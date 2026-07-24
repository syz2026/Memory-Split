from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from cluster.aws.gpu_profile import load_aws_gpu_profile
from msctl.aws_contracts import SNAPSHOT_STEPS
from tests.provider_lifecycle_fixtures import provider_lifecycle


ROOT = Path(__file__).resolve().parents[1]
P5_PROFILE = ROOT / "cluster" / "profiles" / "aws-p5.48xlarge-v3.json"
P6_PROFILE = ROOT / "cluster" / "profiles" / "aws-p6-b300.48xlarge-v3.json"
S3_ROOT = "s3://memorysplit-prod/confirmatory-v3"
RELEASE_SHA256 = "d" * 64
RELEASE_RECEIPT_SHA256 = "e" * 64
DATASET_POINTER_SHA256 = "f" * 64
DATASET_RECEIPT_SHA256 = "0" * 64
DATASET_BUILD_ID = "1" * 64
ORDERED_STREAM_SHA256 = "2" * 64
COHORT_ASSIGNMENT_SHA256 = "3" * 64
PREREGISTRATION_SHA256 = "4" * 64
SEALED_EVALUATION_SHA256 = "5" * 64
SOURCE_COMMIT = "6" * 40
SOURCE_TREE = "7" * 40


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _receipt_value(
    binding,
    *,
    seed: int,
    run_manifest_sha256: str | None = None,
) -> dict[str, object]:
    arms = []
    for arm in ("dense", "split90"):
        snapshots = []
        for step in SNAPSHOT_STEPS:
            digest = hashlib.sha256(
                f"snapshot-{seed}-{arm}-{step}".encode("ascii")
            ).hexdigest()
            snapshots.append(
                {
                    "object": {
                        "bytes": 4096 + step,
                        "sha256": digest,
                        "uri": (
                            f"{S3_ROOT}/snapshots/seed-{seed}/{arm}/"
                            f"step-{step}/sha256/{digest}.pt"
                        ),
                        "version_id": f"snapshot-{arm}-{step}",
                    },
                    "step": step,
                }
            )
        log_digest = hashlib.sha256(
            f"log-{seed}-{arm}".encode("ascii")
        ).hexdigest()
        arms.append(
            {
                "arm": arm,
                "config_fingerprint": hashlib.sha256(
                    f"fingerprint-{arm}".encode("ascii")
                ).hexdigest(),
                "config_sha256": hashlib.sha256(
                    f"config-{arm}".encode("ascii")
                ).hexdigest(),
                "final_step": 13_582,
                "log": {
                    "bytes": 2_048,
                    "sha256": log_digest,
                    "uri": (
                        f"{S3_ROOT}/logs/seed-{seed}/{arm}/"
                        f"sha256/{log_digest}.jsonl"
                    ),
                    "version_id": f"log-{arm}-1",
                },
                "run_id": f"memorysplit-v3-360m-s{seed}-{arm}",
                "snapshots": snapshots,
                "world_size": 4,
            }
        )
    checkpoint_digest = hashlib.sha256(
        f"checkpoint-{seed}".encode("ascii")
    ).hexdigest()
    return {
        "arms": arms,
        "boot_id": binding.boot_id,
        "canary_receipt_sha256": (
            binding.qualification_canary_receipt_sha256
        ),
        "checkpoint_receipt": {
            "sha256": checkpoint_digest,
            "uri": (
                f"{S3_ROOT}/receipts/checkpoints/seed-{seed}/"
                f"sha256/{checkpoint_digest}.json"
            ),
            "version_id": "checkpoint-receipt-1",
        },
        "cohort_id": binding.cohort_id,
        "complete": True,
        "dataset_build_id": DATASET_BUILD_ID,
        "dataset_receipt_sha256": DATASET_RECEIPT_SHA256,
        "environment_receipt_sha256": (
            binding.qualification_environment_receipt_sha256
        ),
        "finalized_at": "2026-07-24T00:00:00Z",
        "hardware_amendment_sha256": binding.hardware_amendment_sha256,
        "instance_id": binding.instance_id,
        "objective_controls_contract_sha256": (
            binding.objective_controls_contract_sha256
        ),
        "ordered_stream_sha256": ORDERED_STREAM_SHA256,
        "profile_id": binding.profile_id,
        "profile_sha256": binding.profile_sha256,
        "provider": binding.provider,
        "provider_selection_sha256": binding.provider_selection_sha256,
        "provider_selection_version_id": (
            binding.provider_selection_version_id
        ),
        "qualification_approval_public_key_sha256": (
            binding.qualification_approval_public_key_sha256
        ),
        "qualification_approval_receipt_sha256": (
            binding.qualification_approval_receipt_sha256
        ),
        "qualification_evidence_sha256": (
            binding.qualification_evidence_sha256
        ),
        "receipt_type": "memorysplit-aws-paired-run-finalization-v3",
        "release_receipt_sha256": RELEASE_RECEIPT_SHA256,
        "release_sha256": RELEASE_SHA256,
        "request_id": "9" * 32,
        "run_manifest_sha256": (
            run_manifest_sha256
            or hashlib.sha256(f"manifest-{seed}".encode("ascii")).hexdigest()
        ),
        "runtime_lock_sha256": binding.runtime_lock_sha256,
        "runtime_sbom_sha256": binding.runtime_sbom_sha256,
        "schema_version": 3,
        "seed": seed,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
    }


def _receipt_ref(value: dict[str, object], *, payload: bytes | None = None):
    from msctl.aws_seed_transition import PriorRunReceiptRef

    body = payload if payload is not None else _canonical(value)
    digest = hashlib.sha256(body).hexdigest()
    return body, PriorRunReceiptRef(
        uri=(
            f"{S3_ROOT}/receipts/runs/seed-{value['seed']}/"
            f"sha256/{digest}.json"
        ),
        sha256=digest,
        version_id="prior-receipt-version-1",
    )


def _admit_arguments(binding) -> dict[str, object]:
    return {
        "binding": binding,
        "release_sha256": RELEASE_SHA256,
        "release_receipt_sha256": RELEASE_RECEIPT_SHA256,
        "dataset_receipt_sha256": DATASET_RECEIPT_SHA256,
        "dataset_build_id": DATASET_BUILD_ID,
        "ordered_stream_sha256": ORDERED_STREAM_SHA256,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "instance_id": binding.instance_id,
    }


@pytest.mark.parametrize("profile_path", [P5_PROFILE, P6_PROFILE])
def test_prior_seed_admission_returns_fourteen_head_only_objects(profile_path):
    from msctl.aws_seed_transition import (
        AdmittedPriorRun,
        PriorEvidenceObject,
        admit_prior_seed_finalization,
    )

    profile = load_aws_gpu_profile(profile_path)
    binding = provider_lifecycle(profile, seed=3).binding
    value = _receipt_value(binding, seed=2)
    payload, ref = _receipt_ref(value)

    admitted = admit_prior_seed_finalization(
        payload,
        ref=ref,
        **_admit_arguments(binding),
    )

    assert isinstance(admitted, AdmittedPriorRun)
    assert admitted.seed == 2
    assert admitted.receipt == ref
    assert len(admitted.evidence) == 14
    assert all(
        isinstance(item, PriorEvidenceObject) for item in admitted.evidence
    )
    assert all(
        item.uri.startswith(S3_ROOT + "/") for item in admitted.evidence
    )
    dense = value["arms"][0]
    split90 = value["arms"][1]
    expected_uris = [
        *[item["object"]["uri"] for item in dense["snapshots"]],
        dense["log"]["uri"],
        *[item["object"]["uri"] for item in split90["snapshots"]],
        split90["log"]["uri"],
        value["checkpoint_receipt"]["uri"],
        ref.uri,
    ]
    assert [item.uri for item in admitted.evidence] == expected_uris
    checkpoint_evidence = admitted.evidence[12]
    assert checkpoint_evidence.bytes is None
    assert checkpoint_evidence.sha256 == (
        value["checkpoint_receipt"]["sha256"]
    )
    receipt_evidence = admitted.evidence[13]
    assert receipt_evidence.bytes == len(payload)
    assert receipt_evidence.sha256 == ref.sha256
    assert receipt_evidence.version_id == ref.version_id
    assert all(
        type(item.bytes) is int and item.bytes > 0
        for item in admitted.evidence[:12]
    )


def test_seed_zero_admission_is_forbidden():
    from msctl.aws_seed_transition import (
        SeedTransitionError,
        admit_prior_seed_finalization,
    )

    profile = load_aws_gpu_profile(P5_PROFILE)
    binding = provider_lifecycle(profile, seed=0).binding
    value = _receipt_value(binding, seed=0)
    payload, ref = _receipt_ref(value)

    with pytest.raises(SeedTransitionError, match="[Ss]eed 0"):
        admit_prior_seed_finalization(
            payload,
            ref=ref,
            **_admit_arguments(binding),
        )


@pytest.mark.parametrize("receipt_seed", [3, 1, 0])
def test_wrong_prior_receipt_seed_fails(receipt_seed):
    from msctl.aws_seed_transition import (
        SeedTransitionError,
        admit_prior_seed_finalization,
    )

    profile = load_aws_gpu_profile(P5_PROFILE)
    binding = provider_lifecycle(profile, seed=3).binding
    value = _receipt_value(binding, seed=receipt_seed)
    payload, ref = _receipt_ref(value)

    with pytest.raises(SeedTransitionError, match="seed"):
        admit_prior_seed_finalization(
            payload,
            ref=ref,
            **_admit_arguments(binding),
        )


_LIFECYCLE_RECEIPT_FIELDS = (
    "canary_receipt_sha256",
    "cohort_id",
    "environment_receipt_sha256",
    "hardware_amendment_sha256",
    "instance_id",
    "objective_controls_contract_sha256",
    "profile_id",
    "profile_sha256",
    "provider",
    "provider_selection_sha256",
    "provider_selection_version_id",
    "qualification_approval_public_key_sha256",
    "qualification_approval_receipt_sha256",
    "qualification_evidence_sha256",
    "runtime_lock_sha256",
    "runtime_sbom_sha256",
)


@pytest.mark.parametrize("field", _LIFECYCLE_RECEIPT_FIELDS)
def test_every_lifecycle_authority_mutation_fails(field):
    from msctl.aws_seed_transition import (
        SeedTransitionError,
        admit_prior_seed_finalization,
    )

    profile = load_aws_gpu_profile(P6_PROFILE)
    binding = provider_lifecycle(profile, seed=5).binding
    value = _receipt_value(binding, seed=4)
    current = value[field]
    if isinstance(current, str) and len(current) == 64:
        value[field] = ("0" if current[0] != "0" else "1") * 64
    elif field == "instance_id":
        value[field] = "i-0fedcba9876543210"
    elif field in {"provider", "profile_id"}:
        # Swap to the other selected provider while keeping the pair
        # mutually consistent, so the mutation is a lifecycle-authority
        # mismatch rather than a parser-shape failure.
        value["provider"] = (
            "aws-p5.48xlarge"
            if value["provider"] == "aws-p6-b300.48xlarge"
            else "aws-p6-b300.48xlarge"
        )
        value["profile_id"] = f"{value['provider']}-v3"
    elif field == "provider_selection_version_id":
        value[field] = "selection-version-2"
    elif field == "cohort_id":
        value[field] = "memorysplit-confirmatory-v3-360m-n10-gcp"
    else:
        value[field] = "mutated-" + str(current)
    payload, ref = _receipt_ref(value)

    with pytest.raises(SeedTransitionError):
        admit_prior_seed_finalization(
            payload,
            ref=ref,
            **_admit_arguments(binding),
        )


def test_boot_only_drift_passes_after_reboot():
    from msctl.aws_seed_transition import admit_prior_seed_finalization

    profile = load_aws_gpu_profile(P5_PROFILE)
    binding = provider_lifecycle(profile, seed=1).binding
    value = _receipt_value(binding, seed=0)
    value["boot_id"] = "87654321-4321-4cba-9abc-0987654321fe"
    assert value["boot_id"] != binding.boot_id
    payload, ref = _receipt_ref(value)

    admitted = admit_prior_seed_finalization(
        payload,
        ref=ref,
        **_admit_arguments(binding),
    )

    assert admitted.seed == 0
    assert len(admitted.evidence) == 14


@pytest.mark.parametrize(
    "mutation",
    [
        "release_sha256",
        "release_receipt_sha256",
        "dataset_receipt_sha256",
        "dataset_build_id",
        "ordered_stream_sha256",
        "source_commit",
        "source_tree",
        "instance_argument",
        "canonical_bytes",
        "receipt_hash",
        "receipt_version",
        "receipt_uri",
        "incomplete",
    ],
)
def test_release_dataset_source_and_object_identity_drift_fails(mutation):
    from msctl.aws_seed_transition import (
        SeedTransitionError,
        admit_prior_seed_finalization,
    )

    profile = load_aws_gpu_profile(P5_PROFILE)
    binding = provider_lifecycle(profile, seed=2).binding
    value = _receipt_value(binding, seed=1)
    arguments = _admit_arguments(binding)
    if mutation in {
        "release_sha256",
        "release_receipt_sha256",
        "dataset_receipt_sha256",
        "dataset_build_id",
        "ordered_stream_sha256",
    }:
        value[mutation] = "a" * 64
    elif mutation in {"source_commit", "source_tree"}:
        value[mutation] = "a" * 40
    elif mutation == "incomplete":
        value["complete"] = False
    payload, ref = _receipt_ref(value)
    if mutation == "instance_argument":
        arguments["instance_id"] = "i-0fedcba9876543210"
    elif mutation == "canonical_bytes":
        payload = payload + b"\n"
    elif mutation == "receipt_hash":
        ref = replace(ref, sha256="b" * 64)
    elif mutation == "receipt_version":
        ref = replace(ref, version_id="prior-receipt-version-2")
        # Version drift alone cannot be detected from bytes; combine with a
        # hash binding to prove the version is still pinned in admission
        # outputs and never silently rewritten.
        payload2, ref2 = _receipt_ref(value)
        admitted = admit_prior_seed_finalization(
            payload2,
            ref=ref2,
            **arguments,
        )
        assert admitted.receipt.version_id == "prior-receipt-version-1"
        assert admitted.evidence[13].version_id == "prior-receipt-version-1"
        admitted_drifted = admit_prior_seed_finalization(
            payload,
            ref=ref,
            **arguments,
        )
        assert admitted_drifted.receipt.version_id == (
            "prior-receipt-version-2"
        )
        return
    elif mutation == "receipt_uri":
        ref = replace(
            ref,
            uri=(
                f"{S3_ROOT}/receipts/runs/seed-1/sha256/"
                + "b" * 64
                + ".json"
            ),
        )

    with pytest.raises(SeedTransitionError):
        admit_prior_seed_finalization(
            payload,
            ref=ref,
            **arguments,
        )


def _prior_triple(seed: int) -> dict[str, str]:
    digest = hashlib.sha256(f"prior-{seed}".encode("ascii")).hexdigest()
    return {
        "uri": (
            f"{S3_ROOT}/receipts/runs/seed-{seed - 1}/sha256/{digest}.json"
        ),
        "sha256": digest,
        "version_id": "prior-receipt-version-1",
    }


def _collection_triple(seed: int) -> dict[str, str]:
    digest = hashlib.sha256(
        f"prior-collection-{seed}".encode("ascii")
    ).hexdigest()
    return {
        "uri": (
            f"{S3_ROOT}/receipts/collections/seed-{seed - 1}/"
            f"sha256/{digest}.json"
        ),
        "sha256": digest,
        "version_id": "prior-collection-version-1",
    }


def _selected_run_state(
    binding,
    *,
    arm: str,
    manifest_sha256: str,
    operation: str = "submit",
    status: str = "Pending",
    command_id: str | None = "cmd-0123456789abcdef0",
    bootstrap_mode: str = "bootstrap",
    bootstrap_receipt_sha256: str | None = None,
    prior_run_receipt: dict[str, str] | None = None,
    prior_collection_receipt: dict[str, str] | None = None,
) -> dict[str, object]:
    seed = binding.seed
    return {
        **binding.to_dict(),
        "schema_version": 2,
        "run_id": f"memorysplit-v3-360m-s{seed}-{arm}",
        "arm": arm,
        "release_sha256": RELEASE_SHA256,
        "release_receipt_sha256": RELEASE_RECEIPT_SHA256,
        "run_manifest_sha256": manifest_sha256,
        "config_sha256": ("a" if arm == "dense" else "b") * 64,
        "dataset_pointer_sha256": DATASET_POINTER_SHA256,
        "dataset_receipt_sha256": DATASET_RECEIPT_SHA256,
        "dataset_build_id": DATASET_BUILD_ID,
        "ordered_stream_sha256": ORDERED_STREAM_SHA256,
        "dataset_verification_sha256": DATASET_RECEIPT_SHA256,
        "environment_receipt_sha256": "8" * 64,
        "cohort_assignment_sha256": COHORT_ASSIGNMENT_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "runtime_sha256": "9" * 64,
        "ami_id": "ami-0123456789abcdef0",
        "container_digest": "sha256:" + "8" * 64,
        "terminate_at": "2099-01-01T00:00:00Z",
        "operation_id": "b" * 64,
        "intent_sha256": "c" * 64,
        "intent_uri": "s3://memorysplit-prod/intents/submit.json",
        "command_id": command_id,
        "operation": operation,
        "status": status,
        "attempt": 1,
        "send_attempted": True,
        "created_at": "2026-07-24T00:00:00Z",
        "updated_at": "2026-07-24T00:00:00Z",
        "bootstrap_mode": bootstrap_mode,
        "bootstrap_receipt_sha256": bootstrap_receipt_sha256,
        "prior_run_receipt": copy.deepcopy(prior_run_receipt),
        "prior_collection_receipt": copy.deepcopy(prior_collection_receipt),
    }


def _selected_pair(
    binding,
    *,
    manifest_sha256: str,
    status: str = "Pending",
    **state_arguments,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    states = [
        _selected_run_state(
            binding,
            arm=arm,
            manifest_sha256=manifest_sha256,
            status=status,
            **state_arguments,
        )
        for arm in ("dense", "split90")
    ]
    journal = {
        "schema_version": 2,
        "provider": binding.provider,
        "run_manifest_sha256": manifest_sha256,
        "operation_id": states[0]["operation_id"],
        "states": states,
    }
    return journal, states


def _manifest_sha(seed: int) -> str:
    return hashlib.sha256(f"selected-manifest-{seed}".encode()).hexdigest()


def test_selected_state_requires_bootstrap_and_prior_receipt_fields(tmp_path):
    from msctl.state import StateStore

    profile = load_aws_gpu_profile(P5_PROFILE)
    binding = provider_lifecycle(profile, seed=0).binding
    manifest_sha256 = _manifest_sha(0)
    journal, states = _selected_pair(binding, manifest_sha256=manifest_sha256)
    store = StateStore(tmp_path / "state")

    with store.locked():
        store.write_aws_pair_transaction(manifest_sha256, journal, states)
        assert store.read_run(str(states[0]["run_id"])) == states[0]

    for mutation in (
        "drop-new-keys",
        "drop-collection-key",
        "unknown-mode",
        "reuse-without-receipt",
        "bootstrap-with-receipt",
        "seed-zero-with-prior",
        "seed-zero-with-prior-collection",
    ):
        journal_copy = copy.deepcopy(journal)
        for state in journal_copy["states"]:
            if mutation == "drop-new-keys":
                state.pop("bootstrap_mode")
                state.pop("bootstrap_receipt_sha256")
                state.pop("prior_run_receipt")
                state.pop("prior_collection_receipt")
            elif mutation == "drop-collection-key":
                state.pop("prior_collection_receipt")
            elif mutation == "unknown-mode":
                state["bootstrap_mode"] = "refresh"
            elif mutation == "reuse-without-receipt":
                state["bootstrap_mode"] = "reuse"
            elif mutation == "bootstrap-with-receipt":
                state["bootstrap_receipt_sha256"] = "b" * 64
            elif mutation == "seed-zero-with-prior":
                state["prior_run_receipt"] = _prior_triple(1)
            elif mutation == "seed-zero-with-prior-collection":
                state["prior_collection_receipt"] = _collection_triple(1)
        target = StateStore(tmp_path / f"state-{mutation}")
        with target.locked(), pytest.raises(Exception) as caught:
            target.write_aws_pair_transaction(
                manifest_sha256,
                journal_copy,
                journal_copy["states"],
            )
        assert getattr(caught.value, "code", None) in {
            "STATE_CORRUPT",
            "SCHEMA_INVALID",
        }, mutation


def test_selected_seed_one_state_requires_exact_prior_triple(tmp_path):
    from msctl.state import StateStore

    profile = load_aws_gpu_profile(P6_PROFILE)
    binding = provider_lifecycle(profile, seed=1).binding
    manifest_sha256 = _manifest_sha(1)
    journal, states = _selected_pair(
        binding,
        manifest_sha256=manifest_sha256,
        prior_run_receipt=_prior_triple(1),
        prior_collection_receipt=_collection_triple(1),
    )
    store = StateStore(tmp_path / "state")
    with store.locked():
        store.write_aws_pair_transaction(manifest_sha256, journal, states)
        stored = store.read_run(str(states[0]["run_id"]))
    assert stored["prior_run_receipt"] == _prior_triple(1)
    assert stored["prior_collection_receipt"] == _collection_triple(1)

    for mutation in (
        "missing",
        "extra-key",
        "bad-hash",
        "null-version",
        "collection-missing",
        "collection-extra-key",
        "collection-bad-hash",
        "collection-null-version",
    ):
        journal_copy = copy.deepcopy(journal)
        for state in journal_copy["states"]:
            if mutation == "missing":
                state["prior_run_receipt"] = None
            elif mutation == "extra-key":
                state["prior_run_receipt"]["bytes"] = 10
            elif mutation == "bad-hash":
                state["prior_run_receipt"]["sha256"] = "not-a-hash"
            elif mutation == "null-version":
                state["prior_run_receipt"]["version_id"] = "null"
            elif mutation == "collection-missing":
                state["prior_collection_receipt"] = None
            elif mutation == "collection-extra-key":
                state["prior_collection_receipt"]["bytes"] = 10
            elif mutation == "collection-bad-hash":
                state["prior_collection_receipt"]["sha256"] = "not-a-hash"
            elif mutation == "collection-null-version":
                state["prior_collection_receipt"]["version_id"] = "null"
        target = StateStore(tmp_path / f"state-{mutation}")
        with target.locked(), pytest.raises(Exception) as caught:
            target.write_aws_pair_transaction(
                manifest_sha256,
                journal_copy,
                journal_copy["states"],
            )
        assert getattr(caught.value, "code", None) in {
            "STATE_CORRUPT",
            "SCHEMA_INVALID",
        }, mutation


def _resume_transition_states(
    states: list[dict[str, object]],
    *,
    terminate_at: str = "2099-02-01T00:00:00Z",
    bootstrap_mode: str = "reuse",
    bootstrap_receipt_sha256: str | None = "b" * 64,
) -> list[dict[str, object]]:
    transitioned = copy.deepcopy(states)
    checkpoint_objects = [
        {
            "arm": arm,
            "bytes": 123,
            "sha256": digest * 64,
            "uri": f"s3://memorysplit-prod/checkpoints/{arm}.pt",
            "version_id": f"{arm}-version-1",
        }
        for arm, digest in (("dense", "1"), ("split90", "2"))
    ]
    for state in transitioned:
        state.update(
            {
                "operation": "resume",
                "operation_id": "3" * 64,
                "intent_sha256": "4" * 64,
                "intent_uri": (
                    "s3://memorysplit-prod/intents/resume.json"
                ),
                "command_id": None,
                "status": "INTENT_PUBLISHED",
                "attempt": 2,
                "send_attempted": False,
                "checkpoint_receipt": {
                    "sha256": "5" * 64,
                    "uri": "s3://memorysplit-prod/receipt.json",
                    "version_id": "receipt-version-1",
                },
                "checkpoint_objects": copy.deepcopy(checkpoint_objects),
                "prior_command_ids": ["cmd-0123456789abcdef0"],
                "terminate_at": terminate_at,
                "bootstrap_mode": bootstrap_mode,
                "bootstrap_receipt_sha256": bootstrap_receipt_sha256,
                "updated_at": "2099-01-02T03:04:05Z",
            }
        )
    return transitioned


def test_selected_resume_transition_updates_lease_scope_only(tmp_path):
    from msctl.state import StateStore

    profile = load_aws_gpu_profile(P5_PROFILE)
    binding = provider_lifecycle(profile, seed=1).binding
    manifest_sha256 = _manifest_sha(1)
    journal, states = _selected_pair(
        binding,
        manifest_sha256=manifest_sha256,
        prior_run_receipt=_prior_triple(1),
        prior_collection_receipt=_collection_triple(1),
        status="Failed",
    )
    store = StateStore(tmp_path / "state")
    with store.locked():
        store.write_aws_pair_transaction(manifest_sha256, journal, states)

    transitioned = _resume_transition_states(states)
    resume_journal = {
        "schema_version": 2,
        "provider": binding.provider,
        "run_manifest_sha256": manifest_sha256,
        "operation_id": transitioned[0]["operation_id"],
        "states": transitioned,
    }
    with store.locked():
        store.write_aws_pair_transaction(
            manifest_sha256,
            resume_journal,
            transitioned,
        )
        stored = store.read_run(str(transitioned[0]["run_id"]))
    assert stored["terminate_at"] == "2099-02-01T00:00:00Z"
    assert stored["bootstrap_mode"] == "reuse"
    assert stored["bootstrap_receipt_sha256"] == "b" * 64
    assert stored["prior_run_receipt"] == _prior_triple(1)
    assert stored["prior_collection_receipt"] == _collection_triple(1)


@pytest.mark.parametrize(
    "field",
    ["prior_run_receipt", "prior_collection_receipt"],
)
def test_selected_resume_transition_keeps_prior_receipts_immutable(
    tmp_path,
    field,
):
    from msctl.state import StateStore

    profile = load_aws_gpu_profile(P5_PROFILE)
    binding = provider_lifecycle(profile, seed=2).binding
    manifest_sha256 = _manifest_sha(2)
    journal, states = _selected_pair(
        binding,
        manifest_sha256=manifest_sha256,
        prior_run_receipt=_prior_triple(2),
        prior_collection_receipt=_collection_triple(2),
        status="Failed",
    )
    store = StateStore(tmp_path / "state")
    with store.locked():
        store.write_aws_pair_transaction(manifest_sha256, journal, states)

    transitioned = _resume_transition_states(states)
    for state in transitioned:
        state[field] = (
            _prior_triple(1)
            if field == "prior_run_receipt"
            else _collection_triple(1)
        )
    resume_journal = {
        "schema_version": 2,
        "provider": binding.provider,
        "run_manifest_sha256": manifest_sha256,
        "operation_id": transitioned[0]["operation_id"],
        "states": transitioned,
    }
    with store.locked(), pytest.raises(Exception) as caught:
        store.write_aws_pair_transaction(
            manifest_sha256,
            resume_journal,
            transitioned,
        )
    assert getattr(caught.value, "code", None) == "STATE_CORRUPT"


def test_selected_same_operation_replay_changes_no_transition_fields(tmp_path):
    from msctl.state import StateStore

    profile = load_aws_gpu_profile(P5_PROFILE)
    binding = provider_lifecycle(profile, seed=0).binding
    manifest_sha256 = _manifest_sha(0)
    journal, states = _selected_pair(binding, manifest_sha256=manifest_sha256)
    store = StateStore(tmp_path / "state")
    with store.locked():
        store.write_aws_pair_transaction(manifest_sha256, journal, states)

    for field, value in (
        ("bootstrap_mode", "reuse"),
        ("bootstrap_receipt_sha256", "b" * 64),
        ("terminate_at", "2099-02-01T00:00:00Z"),
        ("prior_collection_receipt", _collection_triple(1)),
    ):
        replayed = copy.deepcopy(journal)
        for state in replayed["states"]:
            state[field] = value
            if field == "bootstrap_mode":
                state["bootstrap_receipt_sha256"] = "b" * 64
            state["updated_at"] = "2099-01-02T03:04:05Z"
        with store.locked(), pytest.raises(Exception) as caught:
            store.write_aws_pair_transaction(
                manifest_sha256,
                replayed,
                replayed["states"],
            )
        assert getattr(caught.value, "code", None) == "STATE_CORRUPT", field


def test_legacy_v3_resume_transition_still_reuses_stored_deadline(tmp_path):
    from tests.test_aws_paired_state import _transition_to_resume, _v3_pair

    fixture = _v3_pair(tmp_path, status="Failed")
    proposed = _transition_to_resume(fixture)
    assert {
        state["terminate_at"] for state in proposed
    } == {fixture.terminate_at}

    drifted = copy.deepcopy(proposed)
    for state in drifted:
        state["terminate_at"] = "2099-02-01T00:00:00Z"
        state["operation_id"] = "6" * 64
        state["intent_sha256"] = "7" * 64
        state["intent_uri"] = "s3://bucket/resume-intent-2.json"
        state["updated_at"] = "2099-01-03T00:00:00Z"
    journal = {
        "schema_version": 2,
        "provider": "aws-p5.48xlarge",
        "run_manifest_sha256": fixture.manifest.sha256,
        "operation_id": drifted[0]["operation_id"],
        "states": drifted,
    }
    with fixture.store.locked(), pytest.raises(Exception) as caught:
        fixture.store.write_aws_pair_transaction(
            fixture.manifest.sha256,
            journal,
            drifted,
        )
    assert getattr(caught.value, "code", None) == "STATE_CORRUPT"


def test_read_all_aws_pairs_returns_every_validated_journal(tmp_path):
    from msctl.errors import MsctlError
    from msctl.state import StateStore

    profile = load_aws_gpu_profile(P5_PROFILE)
    store = StateStore(tmp_path / "state")
    written: dict[str, dict[str, object]] = {}
    for seed in (0, 1):
        binding = provider_lifecycle(profile, seed=seed).binding
        manifest_sha256 = _manifest_sha(seed)
        journal, states = _selected_pair(
            binding,
            manifest_sha256=manifest_sha256,
            status="Success",
            prior_run_receipt=_prior_triple(seed) if seed else None,
            prior_collection_receipt=(
                _collection_triple(seed) if seed else None
            ),
        )
        with store.locked():
            store.write_aws_pair_transaction(
                manifest_sha256,
                journal,
                states,
            )
        written[manifest_sha256] = journal

    with pytest.raises(MsctlError) as caught:
        store.read_all_aws_pairs()
    assert caught.value.code == "UNSAFE_STATE"

    with store.locked():
        journals = store.read_all_aws_pairs()
    assert set(journals) == set(written)
    for manifest_sha256, journal in journals.items():
        assert journal["run_manifest_sha256"] == manifest_sha256
        assert len(journal["states"]) == 2

    corrupt = (
        tmp_path / "state" / "intents" / ("aws-" + "f" * 64 + ".json")
    )
    corrupt.write_text("{}", encoding="ascii")
    with store.locked(), pytest.raises(MsctlError) as caught:
        store.read_all_aws_pairs()
    assert caught.value.code == "STATE_CORRUPT"


BOOT_ID = "12345678-1234-4abc-8def-1234567890ab"
INSTANCE_ID = "i-0123456789abcdef0"
IMAGE_DIGEST = "sha256:" + "8" * 64
IMAGE = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/memorysplit"
    f"@{IMAGE_DIGEST}"
)


def _bootstrap_profile():
    return SimpleNamespace(
        provider="aws-p5.48xlarge",
        profile_id="aws-p5.48xlarge-v3",
        instance_type="p5.48xlarge",
        sha256="7" * 64,
        scratch_root="/mnt/memorysplit",
        instance_store_devices=8,
        instance_store_device_bytes=3_800_000_000_000,
        instance_store_model="Amazon EC2 NVMe Instance Storage",
        raid_level=0,
    )


def _bootstrap_runtime(*, uid: int | None = None, gid: int | None = None):
    import os

    return SimpleNamespace(
        region="us-east-1",
        s3_root=S3_ROOT,
        ami_id="ami-0123456789abcdef0",
        container_image=IMAGE,
        container_digest=IMAGE_DIGEST,
        uid=os.getuid() if uid is None else uid,
        gid=os.getgid() if gid is None else gid,
    )


def _bootstrap_receipt_value(profile, runtime) -> dict[str, object]:
    return {
        "account_id": "123456789012",
        "ami_id": runtime.ami_id,
        "boot_id": BOOT_ID,
        "code_commit": SOURCE_COMMIT,
        "cohort_assignment_sha256": COHORT_ASSIGNMENT_SHA256,
        "container_image": runtime.container_image,
        "container_digest": runtime.container_digest,
        "corpus_build_id": DATASET_BUILD_ID,
        "corpus_ordered_stream_sha256": ORDERED_STREAM_SHA256,
        "corpus_receipt_sha256": "0" * 64,
        "durable_upload_verified": True,
        "instance_id": INSTANCE_ID,
        "instance_store": {
            "device_bytes": profile.instance_store_device_bytes,
            "devices": profile.instance_store_devices,
            "model": profile.instance_store_model,
            "raid_level": profile.raid_level,
        },
        "instance_type": profile.instance_type,
        "profile_sha256": profile.sha256,
        "provider": profile.provider,
        "receipt_type": "aws-p5-bootstrap",
        "region": runtime.region,
        "release_members_sha256": "6" * 64,
        "release_root": f"releases/{RELEASE_SHA256}",
        "release_sha256": RELEASE_SHA256,
        "role_arn": (
            "arn:aws:sts::123456789012:assumed-role/memorysplit-p5/i"
        ),
        "role_name": "memorysplit-p5",
        "runtime_gid": runtime.gid,
        "runtime_uid": runtime.uid,
        "schema_version": 2,
        "scratch_root": profile.scratch_root,
    }


def _verify_reuse_fixture(tmp_path: Path):
    import os

    profile = _bootstrap_profile()
    scratch = tmp_path / "scratch"
    (scratch / "staging").mkdir(parents=True)
    (scratch / "releases" / RELEASE_SHA256).mkdir(parents=True)
    dataset = scratch / "dataset"
    dataset.mkdir()
    dataset_receipt = dataset / "receipt.json"
    dataset_receipt.write_bytes(b'{"build":"canonical"}\n')
    os.chmod(scratch, 0o700)
    metadata = os.stat(scratch)
    runtime = _bootstrap_runtime(uid=metadata.st_uid, gid=metadata.st_gid)
    value = _bootstrap_receipt_value(profile, runtime)
    value["corpus_receipt_sha256"] = hashlib.sha256(
        dataset_receipt.read_bytes()
    ).hexdigest()
    receipt_path = scratch / "staging" / "bootstrap-receipt.json"
    receipt_path.write_bytes(_canonical(value))
    return SimpleNamespace(
        profile=profile,
        runtime=runtime,
        scratch=scratch,
        dataset_receipt=dataset_receipt,
        receipt_path=receipt_path,
        value=value,
        receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    )


def _scratch_snapshot(scratch: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(scratch)): path.read_bytes()
        for path in sorted(scratch.rglob("*"))
        if path.is_file()
    }


def test_verify_bootstrap_reuse_accepts_exact_receipt_without_mutation(
    tmp_path,
):
    from cluster.aws.p5.bootstrap import verify_bootstrap_reuse

    fixture = _verify_reuse_fixture(tmp_path)
    before = _scratch_snapshot(fixture.scratch)

    result = verify_bootstrap_reuse(
        profile=fixture.profile,
        runtime=fixture.runtime,
        receipt_path=fixture.receipt_path,
        expected_receipt_sha256=fixture.receipt_sha256,
        metadata_get=lambda path: (
            INSTANCE_ID if path == "meta-data/instance-id" else None
        ),
        boot_id_get=lambda: BOOT_ID,
        scratch_root=fixture.scratch,
        is_mounted=lambda path: True,
    )

    assert result["ok"] is True
    assert result["mode"] == "reuse"
    assert result["receipt_sha256"] == fixture.receipt_sha256
    assert _scratch_snapshot(fixture.scratch) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "receipt-hash",
        "boot-drift",
        "instance-drift",
        "not-mounted",
        "scratch-owner",
        "scratch-mode",
        "release-root-missing",
        "release-root-symlink",
        "dataset-hash",
        "durable-not-verified",
        "receipt-fields",
        "receipt-symlink",
    ],
)
def test_verify_bootstrap_reuse_rejects_each_drift_without_mutation(
    tmp_path,
    mutation,
):
    import os

    from cluster.aws.p5.bootstrap import (
        BootstrapError,
        verify_bootstrap_reuse,
    )

    fixture = _verify_reuse_fixture(tmp_path)
    boot_id = BOOT_ID
    instance_id = INSTANCE_ID
    mounted = True
    expected = fixture.receipt_sha256
    if mutation == "receipt-hash":
        expected = "b" * 64
    elif mutation == "boot-drift":
        boot_id = "87654321-4321-4cba-9abc-0987654321fe"
    elif mutation == "instance-drift":
        instance_id = "i-0fedcba9876543210"
    elif mutation == "not-mounted":
        mounted = False
    elif mutation == "scratch-owner":
        fixture.value["runtime_uid"] = fixture.runtime.uid + 1
        fixture.receipt_path.write_bytes(_canonical(fixture.value))
        expected = hashlib.sha256(
            fixture.receipt_path.read_bytes()
        ).hexdigest()
    elif mutation == "scratch-mode":
        os.chmod(fixture.scratch, 0o755)
    elif mutation == "release-root-missing":
        (fixture.scratch / "releases" / RELEASE_SHA256).rmdir()
    elif mutation == "release-root-symlink":
        (fixture.scratch / "releases" / RELEASE_SHA256).rmdir()
        (fixture.scratch / "releases" / RELEASE_SHA256).symlink_to(
            fixture.scratch / "staging"
        )
    elif mutation == "dataset-hash":
        fixture.dataset_receipt.write_bytes(b'{"build":"tampered"}\n')
    elif mutation == "durable-not-verified":
        fixture.value["durable_upload_verified"] = False
        fixture.receipt_path.write_bytes(_canonical(fixture.value))
        expected = hashlib.sha256(
            fixture.receipt_path.read_bytes()
        ).hexdigest()
    elif mutation == "receipt-fields":
        fixture.value.pop("release_root")
        fixture.receipt_path.write_bytes(_canonical(fixture.value))
        expected = hashlib.sha256(
            fixture.receipt_path.read_bytes()
        ).hexdigest()
    elif mutation == "receipt-symlink":
        real = fixture.receipt_path.with_name("real-receipt.json")
        fixture.receipt_path.rename(real)
        fixture.receipt_path.symlink_to(real)
    before = _scratch_snapshot(fixture.scratch)

    with pytest.raises(BootstrapError):
        verify_bootstrap_reuse(
            profile=fixture.profile,
            runtime=fixture.runtime,
            receipt_path=fixture.receipt_path,
            expected_receipt_sha256=expected,
            metadata_get=lambda path: (
                instance_id if path == "meta-data/instance-id" else None
            ),
            boot_id_get=lambda: boot_id,
            scratch_root=fixture.scratch,
            is_mounted=lambda path: mounted,
        )

    assert _scratch_snapshot(fixture.scratch) == before


def test_bootstrap_cli_verify_reuse_requires_exact_flags(tmp_path, capsys):
    from cluster.aws.p5.bootstrap import main

    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b"{}\n")

    assert (
        main(
            [
                "--verify-reuse",
                "--receipt",
                str(receipt),
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["ok"] is False

    assert (
        main(
            [
                "--verify-reuse",
                "--receipt",
                str(receipt),
                "--expected-receipt-sha256",
                "a" * 64,
                "--apply",
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["ok"] is False

    assert main([]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_bootstrap_receipt_field_contract_matches_builder():
    import inspect

    from cluster.aws.p5.bootstrap import (
        BOOTSTRAP_RECEIPT_FIELDS,
        build_bootstrap_receipt,
    )

    source = inspect.getsource(build_bootstrap_receipt)
    for field in BOOTSTRAP_RECEIPT_FIELDS:
        assert f'"{field}"' in source
    assert len(BOOTSTRAP_RECEIPT_FIELDS) == 27


class _ScriptedRunner:
    def __init__(self, *outputs: object) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[list[str], str]] = []

    def run_json(self, argv, *, operation: str):
        self.calls.append((list(argv), operation))
        if not self.outputs:
            raise AssertionError(f"unexpected AWS call: {operation}")
        output = self.outputs.pop(0)
        if callable(output):
            output = output(argv)
        if isinstance(output, Exception):
            raise output
        return output


def _download(payload: bytes, output: dict[str, object]):
    def _respond(argv: list[str]):
        destination = argv[argv.index("--checksum-mode") + 2]
        Path(destination).write_bytes(payload)
        return output

    return _respond


def _selected_backend(tmp_path, monkeypatch, *, seed: int, profile_path=P5_PROFILE):
    import msctl.aws_p5 as module

    profile = load_aws_gpu_profile(profile_path)
    monkeypatch.setattr(
        module,
        "admit_provider_lifecycle",
        lambda **kwargs: provider_lifecycle(profile, seed=kwargs["seed"]),
        raising=False,
    )
    lifecycle = provider_lifecycle(profile, seed=seed)
    runner = _ScriptedRunner()
    environment = {
        "AWS_REGION": "us-east-1",
        "LANG": "C",
        "LC_ALL": "C",
        "MS_S3_ROOT": S3_ROOT,
        "MS_AWS_AMI_ID": "ami-0123456789abcdef0",
        "MS_CONTAINER_DIGEST": IMAGE_DIGEST,
        "MS_CONTAINER_IMAGE": IMAGE,
        "MS_RUNTIME_GID": "10001",
        "MS_RUNTIME_UID": "10001",
    }
    backend = module.AwsP5Backend.from_authenticated_selection(
        authority_root=tmp_path / "authority",
        repo_root=ROOT,
        runtime_lock_path=tmp_path / "runtime-lock.json",
        runtime_evidence_path=tmp_path / "runtime-evidence.json",
        runtime_sbom_path=tmp_path / "runtime-sbom.json",
        objective_controls_amendment_path=(
            ROOT / "configs/objective-controls-amendment-v3.yaml"
        ),
        selection_store=object(),
        account_id=lifecycle.binding.account_id,
        instance_id=lifecycle.binding.instance_id,
        boot_id=lifecycle.binding.boot_id,
        seed=seed,
        expected_selection_version_id=(
            lifecycle.binding.provider_selection_version_id
        ),
        selection_identity_verifier=object(),
        qualification_approval_verifier=object(),
        trusted_qualification_public_key_sha256=(
            lifecycle.binding.qualification_approval_public_key_sha256
        ),
        runtime_environment=environment,
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/MemorySplitSelected"
        ),
        state_root=tmp_path / "state",
        runner=runner,
    )
    return backend, lifecycle, runner


def _selected_flow_manifest(binding):
    seed = binding.seed
    runs = tuple(
        SimpleNamespace(
            run_id=f"memorysplit-v3-360m-s{seed}-{arm}",
            arm=arm,
            seed=seed,
            config=f"configs/360m-v3/{arm}-s{seed}.yaml",
            config_sha256=("a" if arm == "dense" else "b") * 64,
        )
        for arm in ("dense", "split90")
    )
    return SimpleNamespace(
        **binding.to_dict(),
        schema_version=3,
        release_sha256=RELEASE_SHA256,
        release_receipt_sha256=RELEASE_RECEIPT_SHA256,
        dataset_pointer_sha256=DATASET_POINTER_SHA256,
        dataset_receipt_sha256=DATASET_RECEIPT_SHA256,
        dataset_build_id=DATASET_BUILD_ID,
        ordered_stream_sha256=ORDERED_STREAM_SHA256,
        cohort_assignment_sha256=COHORT_ASSIGNMENT_SHA256,
        preregistration_sha256=PREREGISTRATION_SHA256,
        sealed_evaluation_release_sha256=SEALED_EVALUATION_SHA256,
        source_commit=SOURCE_COMMIT,
        source_tree=SOURCE_TREE,
        sha256=_manifest_sha(seed),
        runs=runs,
    )


def _selected_release(manifest):
    return SimpleNamespace(
        provider=manifest.provider,
        archive_sha256=manifest.release_sha256,
        receipt_sha256=manifest.release_receipt_sha256,
        members_sha256="6" * 64,
        source_commit=manifest.source_commit,
        source_tree=manifest.source_tree,
    )


def _selected_evidence(manifest, *, boot_id: str | None = None):
    return {
        "dataset_pointer_sha256": manifest.dataset_pointer_sha256,
        "dataset_verification_sha256": manifest.dataset_receipt_sha256,
        "environment_receipt_sha256": "8" * 64,
        "instance_id": manifest.instance_id,
        "boot_id": boot_id or manifest.boot_id,
    }


def _fresh_deadline() -> str:
    return (
        (datetime.now(UTC) + timedelta(hours=6))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _controller_bootstrap_receipt(manifest, backend) -> dict[str, object]:
    return {
        "account_id": manifest.account_id,
        "ami_id": backend.runtime.ami_id,
        "boot_id": manifest.boot_id,
        "code_commit": manifest.source_commit,
        "cohort_assignment_sha256": manifest.cohort_assignment_sha256,
        "container_image": backend.runtime.container_image,
        "container_digest": backend.runtime.container_digest,
        "corpus_build_id": manifest.dataset_build_id,
        "corpus_ordered_stream_sha256": manifest.ordered_stream_sha256,
        "corpus_receipt_sha256": manifest.dataset_receipt_sha256,
        "durable_upload_verified": True,
        "instance_id": manifest.instance_id,
        "instance_store": {
            "device_bytes": backend.profile.instance_store_device_bytes,
            "devices": backend.profile.instance_store_devices,
            "model": backend.profile.instance_store_model,
            "raid_level": backend.profile.raid_level,
        },
        "instance_type": backend.profile.instance_type,
        "profile_sha256": backend.profile.sha256,
        "provider": backend.profile.provider,
        "receipt_type": "aws-p5-bootstrap",
        "region": backend.runtime.region,
        "release_members_sha256": "6" * 64,
        "release_root": f"releases/{manifest.release_sha256}",
        "release_sha256": manifest.release_sha256,
        "role_arn": (
            "arn:aws:sts::123456789012:assumed-role/memorysplit-p5/i"
        ),
        "role_name": "memorysplit-p5",
        "runtime_gid": 10_001,
        "runtime_uid": 10_001,
        "schema_version": 2,
        "scratch_root": backend.profile.scratch_root,
    }


def _selected_identity_instance(manifest, backend, *, bound: bool):
    tags = {
        "provider": manifest.provider,
        "cohort_sha256": manifest.cohort_assignment_sha256,
        "release_sha256": manifest.release_sha256,
        "dataset_sha256": manifest.dataset_receipt_sha256,
        "profile_sha256": backend.profile.sha256,
        "runtime_sha256": backend._runtime_sha256(),
        "container_digest": backend.runtime.container_digest,
        "selection_sha256": manifest.provider_selection_sha256,
        "selection_version_id": manifest.provider_selection_version_id,
    }
    if not bound:
        tags = {key: None for key in tags}
    return {
        "instance_id": manifest.instance_id,
        "instance_type": backend.profile.instance_type,
        "state": "running",
        "instance_profile_arn": (
            "arn:aws:iam::123456789012:instance-profile/MemorySplitSelected"
        ),
        "ami_id": backend.runtime.ami_id,
        **tags,
    }


def _prior_receipt_download(payload: bytes, ref):
    import base64

    return _download(
        payload,
        {
            "receipt": {
                "checksum_sha256": base64.b64encode(
                    bytes.fromhex(ref.sha256)
                ).decode("ascii"),
                "version_id": ref.version_id,
            }
        },
    )


def _prior_collection_bits(prior_value, prior_payload, prior_ref):
    """Build one valid prior collection receipt for a prior finalization."""

    from tests.test_aws_collect import _collection_ref, _collection_value

    collection_value = _collection_value(
        prior_value,
        prior_payload,
        prior_ref,
    )
    collection_payload, collection_ref = _collection_ref(collection_value)
    return collection_value, collection_payload, collection_ref


def _collection_checkpoint_head_outputs(collection_value) -> list[object]:
    import base64

    return [
        {
            "object": {
                "checksum_sha256": base64.b64encode(
                    bytes.fromhex(row["sha256"])
                ).decode("ascii"),
                "content_length": row["bytes"],
                "version_id": row["version_id"],
            }
        }
        for row in collection_value["objects"][12:14]
    ]


def _evidence_head_outputs(admitted) -> list[object]:
    import base64

    outputs = []
    for item in admitted.evidence:
        outputs.append(
            {
                "object": {
                    "checksum_sha256": base64.b64encode(
                        bytes.fromhex(item.sha256)
                    ).decode("ascii"),
                    "content_length": (
                        item.bytes if item.bytes is not None else 1
                    ),
                    "version_id": item.version_id,
                }
            }
        )
    return outputs


def _submit_tail_outputs(manifest, backend, *, bound_discovery: bool):
    checksum_box: dict[str, str] = {}

    def _put(argv: list[str]):
        checksum = argv[argv.index("--checksum-sha256") + 1]
        checksum_box["value"] = checksum
        return {
            "object": {
                "checksum_sha256": checksum,
                "version_id": "intent-version-1",
            }
        }

    def _head(argv: list[str]):
        body_path = None
        del body_path
        digest = checksum_box["value"]
        intent_path = sorted(
            Path(backend.state_root).glob("intent-*.json")
        )[-1]
        payload = intent_path.read_bytes()
        return {
            "object": {
                "checksum_sha256": digest,
                "content_length": len(payload),
                "metadata": {
                    "operation-id": json.loads(payload)["operation_id"],
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                "version_id": "intent-version-1",
            }
        }

    return [
        {
            "instances": (
                [
                    _selected_identity_instance(
                        manifest,
                        backend,
                        bound=True,
                    )
                ]
                if bound_discovery
                else []
            )
        },
        {
            "instances": [
                _selected_identity_instance(
                    manifest,
                    backend,
                    bound=bound_discovery,
                )
            ]
        },
        {
            "managed_instances": [
                {
                    "instance_id": manifest.instance_id,
                    "ping_status": "Online",
                }
            ]
        },
        _argv_document_listing(),
        {
            "instances": [
                _selected_identity_instance(
                    manifest,
                    backend,
                    bound=bound_discovery,
                )
            ]
        },
        {},
        {},
        {
            "instances": [
                _selected_identity_instance(manifest, backend, bound=True)
            ]
        },
        {
            "attribute": {
                "instance_id": manifest.instance_id,
                "shutdown_behavior": "terminate",
            }
        },
        _put,
        _head,
        {"command": {"command_id": "cmd-0123456789abcdef0"}},
    ]


def _argv_document_listing() -> dict[str, object]:
    from msctl.aws_argv import ARGV_DOCUMENT_NAME, ARGV_DOCUMENT_SHA256

    return {
        "documents": [
            {
                "name": ARGV_DOCUMENT_NAME,
                "hash": ARGV_DOCUMENT_SHA256,
                "status": "Active",
            }
        ]
    }


def _approval_recorder(record: dict[str, object], runner: _ScriptedRunner):
    def _approve(**kwargs):
        record["resources"] = copy.deepcopy(kwargs.get("resources"))
        record["calls_at_approval"] = len(runner.calls)
        return {}

    return _approve


def test_selected_submit_seed_one_full_flow_and_identity_tags(
    tmp_path,
    monkeypatch,
):
    from msctl.aws_argv import _validate_intent
    from msctl.state import StateStore

    backend, lifecycle, runner = _selected_backend(
        tmp_path,
        monkeypatch,
        seed=1,
    )
    manifest = _selected_flow_manifest(lifecycle.binding)
    release = _selected_release(manifest)
    evidence = _selected_evidence(manifest)
    terminate_at = _fresh_deadline()
    approval_record: dict[str, object] = {}
    backend.approval_verifier = _approval_recorder(approval_record, runner)

    prior_binding = provider_lifecycle(
        load_aws_gpu_profile(P5_PROFILE),
        seed=1,
    ).binding
    prior_value = _receipt_value(prior_binding, seed=0)
    prior_payload, prior_ref = _receipt_ref(prior_value)
    from msctl.aws_seed_transition import admit_prior_seed_finalization

    admitted = admit_prior_seed_finalization(
        prior_payload,
        ref=prior_ref,
        **_admit_arguments(prior_binding),
    )
    collection_value, collection_payload, collection_ref = (
        _prior_collection_bits(prior_value, prior_payload, prior_ref)
    )

    prior_state_binding = provider_lifecycle(
        load_aws_gpu_profile(P5_PROFILE),
        seed=0,
    ).binding
    prior_journal, prior_states = _selected_pair(
        prior_state_binding,
        manifest_sha256=_manifest_sha(0),
        status="Success",
    )
    store = StateStore(tmp_path / "state")
    with store.locked():
        store.write_aws_pair_transaction(
            _manifest_sha(0),
            prior_journal,
            prior_states,
        )

    from msctl.errors import MsctlError

    runner.outputs = [
        _prior_receipt_download(prior_payload, prior_ref),
        *_evidence_head_outputs(admitted),
        _prior_receipt_download(collection_payload, collection_ref),
        *_collection_checkpoint_head_outputs(collection_value),
        MsctlError("AWS_COMMAND_FAILED", "no bootstrap receipt"),
        *_submit_tail_outputs(manifest, backend, bound_discovery=False),
    ]

    result = backend.submit(
        release=release,
        manifest=manifest,
        instance_id=manifest.instance_id,
        terminate_at=terminate_at,
        approval_path=tmp_path / "approval.json",
        apply=True,
        evidence=evidence,
        bootstrap_mode="bootstrap",
        prior_run_receipt={
            "uri": prior_ref.uri,
            "sha256": prior_ref.sha256,
            "version_id": prior_ref.version_id,
        },
        prior_collection_receipt={
            "uri": collection_ref.uri,
            "sha256": collection_ref.sha256,
            "version_id": collection_ref.version_id,
        },
    )

    assert result["submitted"] == 1
    assert result["status"] == "Pending"
    resources = approval_record["resources"]
    assert resources["bootstrap_mode"] == "bootstrap"
    assert resources["prior_run_receipt"] == {
        "uri": prior_ref.uri,
        "sha256": prior_ref.sha256,
        "version_id": prior_ref.version_id,
    }
    assert resources["prior_collection_receipt"] == {
        "uri": collection_ref.uri,
        "sha256": collection_ref.sha256,
        "version_id": collection_ref.version_id,
    }
    assert resources["terminate_at"] == terminate_at
    assert approval_record["calls_at_approval"] == 0

    head_calls = [
        argv
        for argv, operation in runner.calls
        if "head-object" in argv and "--version-id" in argv
        and operation == "verify prior evidence object"
    ]
    assert len(head_calls) == 14
    checkpoint_head_calls = [
        argv
        for argv, operation in runner.calls
        if "head-object" in argv and "--version-id" in argv
        and operation == "verify prior collection checkpoint"
    ]
    assert len(checkpoint_head_calls) == 2
    collection_get_calls = [
        argv
        for argv, operation in runner.calls
        if "get-object" in argv
        and operation == "fetch prior seed collection"
    ]
    assert len(collection_get_calls) == 1
    ordered_operations = [operation for _argv, operation in runner.calls]
    assert ordered_operations.index("fetch prior seed collection") > (
        max(
            index
            for index, operation in enumerate(ordered_operations)
            if operation == "verify prior evidence object"
        )
    )
    assert ordered_operations.index("resolve bootstrap receipt") > (
        max(
            index
            for index, operation in enumerate(ordered_operations)
            if operation == "verify prior collection checkpoint"
        )
    )
    assert not any(
        "get-object" in argv and "snapshots/" in " ".join(argv)
        for argv, _operation in runner.calls
    )

    tag_calls = [
        argv for argv, _operation in runner.calls if "create-tags" in argv
    ]
    assert len(tag_calls) == 1
    tags = json.loads(tag_calls[0][tag_calls[0].index("--tags") + 1])
    tag_names = [tag["Key"] for tag in tags]
    assert tag_names == [
        "MemorySplitProvider",
        "MemorySplitCohortSHA256",
        "MemorySplitReleaseSHA256",
        "MemorySplitDatasetSHA256",
        "MemorySplitProfileSHA256",
        "MemorySplitRuntimeSHA256",
        "MemorySplitContainerDigest",
        "MemorySplitSelectionSHA256",
        "MemorySplitSelectionVersionId",
    ]
    assert "MemorySplitSeed" not in tag_names
    assert "MemorySplitRunManifestSHA256" not in tag_names
    assert "MemorySplitTerminateAt" not in tag_names

    discovery_calls = [
        argv
        for argv, operation in runner.calls
        if operation == "selected instance discovery"
    ]
    assert len(discovery_calls) == 1
    assert "describe-instances" in discovery_calls[0]
    filters = " ".join(discovery_calls[0])
    assert "MemorySplitSelectionSHA256" in filters
    assert "MemorySplitSeed" not in filters

    intent_path = sorted(Path(backend.state_root).glob("intent-*.json"))[-1]
    payload = intent_path.read_bytes()
    intent = _validate_intent(
        payload,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert intent["schema_version"] == 3
    assert intent["bootstrap"] == {
        "mode": "bootstrap",
        "receipt_sha256": None,
    }
    from msctl.jsonutil import canonical_sha256

    assert intent["lease_unit"] == (
        "memorysplit-auto-terminate-"
        + canonical_sha256(
            {
                "attempt": 1,
                "operation": "submit",
                "run_manifest_sha256": manifest.sha256,
                "seed": manifest.seed,
                "terminate_at": terminate_at,
            }
        )
    )
    names = [step["name"] for step in intent["steps"]]
    assert names == [
        "reset-termination-leases",
        "auto-termination",
        "prepare-aws-private-home",
        "bootstrap",
        "build-launcher-manifest",
        "paired-launch",
    ]

    with store.locked():
        stored = store.read_run(str(manifest.runs[0].run_id))
    assert stored["bootstrap_mode"] == "bootstrap"
    assert stored["bootstrap_receipt_sha256"] is None
    assert stored["prior_run_receipt"] == {
        "uri": prior_ref.uri,
        "sha256": prior_ref.sha256,
        "version_id": prior_ref.version_id,
    }
    assert stored["prior_collection_receipt"] == {
        "uri": collection_ref.uri,
        "sha256": collection_ref.sha256,
        "version_id": collection_ref.version_id,
    }
    assert stored["terminate_at"] == terminate_at


def test_selected_seed_zero_submit_resolves_reuse_and_forbids_prior(
    tmp_path,
    monkeypatch,
):
    from msctl.aws_argv import _validate_intent

    backend, lifecycle, runner = _selected_backend(
        tmp_path,
        monkeypatch,
        seed=0,
    )
    manifest = _selected_flow_manifest(lifecycle.binding)
    release = _selected_release(manifest)
    evidence = _selected_evidence(manifest)
    terminate_at = _fresh_deadline()
    backend.approval_verifier = lambda **_kwargs: {}

    with pytest.raises(Exception) as caught:
        backend.submit(
            release=release,
            manifest=manifest,
            instance_id=manifest.instance_id,
            terminate_at=terminate_at,
            approval_path=tmp_path / "approval.json",
            apply=True,
            evidence=evidence,
            bootstrap_mode="reuse",
            prior_run_receipt=_prior_triple(1),
            prior_collection_receipt=_collection_triple(1),
        )
    assert getattr(caught.value, "code", None) == "SEED_TRANSITION_BLOCKED"
    assert runner.calls == []

    with pytest.raises(Exception) as caught:
        backend.submit(
            release=release,
            manifest=manifest,
            instance_id=manifest.instance_id,
            terminate_at=terminate_at,
            approval_path=tmp_path / "approval.json",
            apply=True,
            evidence=evidence,
            bootstrap_mode="reuse",
            prior_run_receipt=None,
            prior_collection_receipt=_collection_triple(1),
        )
    assert getattr(caught.value, "code", None) == "SEED_TRANSITION_BLOCKED"
    assert runner.calls == []

    receipt_payload = _canonical(
        _controller_bootstrap_receipt(manifest, backend)
    )
    receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
    runner.outputs = [
        _download(
            receipt_payload,
            {
                "receipt": {
                    "content_length": len(receipt_payload),
                    "version_id": "bootstrap-receipt-1",
                }
            },
        ),
        *_submit_tail_outputs(manifest, backend, bound_discovery=False),
    ]

    result = backend.submit(
        release=release,
        manifest=manifest,
        instance_id=manifest.instance_id,
        terminate_at=terminate_at,
        approval_path=tmp_path / "approval.json",
        apply=True,
        evidence=evidence,
        bootstrap_mode="reuse",
        prior_run_receipt=None,
    )

    assert result["submitted"] == 1
    intent_path = sorted(Path(backend.state_root).glob("intent-*.json"))[-1]
    payload = intent_path.read_bytes()
    intent = _validate_intent(
        payload,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert intent["bootstrap"] == {
        "mode": "reuse",
        "receipt_sha256": receipt_sha256,
    }
    names = [step["name"] for step in intent["steps"]]
    assert names == [
        "reset-termination-leases",
        "auto-termination",
        "verify-bootstrap-reuse",
        "build-launcher-manifest",
        "paired-launch",
    ]
    verify_step = intent["steps"][2]["argv"]
    assert "--verify-reuse" in verify_step
    assert receipt_sha256 in verify_step


def test_selected_submit_gates_fail_before_any_aws_call_or_state(
    tmp_path,
    monkeypatch,
):
    from msctl.state import StateStore

    for case, other_seed, other_status, expected_code in (
        ("active-pair", 0, "InProgress", "SEQUENTIAL_PAIR_ACTIVE"),
        ("active-later-seed", 3, "InProgress", "SEQUENTIAL_PAIR_ACTIVE"),
        ("terminal-later-seed", 2, "Success", "SEED_TRANSITION_BLOCKED"),
    ):
        root = tmp_path / case
        backend, lifecycle, runner = _selected_backend(
            root,
            monkeypatch,
            seed=1,
        )
        manifest = _selected_flow_manifest(lifecycle.binding)
        release = _selected_release(manifest)
        backend.approval_verifier = lambda **_kwargs: {}
        other_binding = provider_lifecycle(
            load_aws_gpu_profile(P5_PROFILE),
            seed=other_seed,
        ).binding
        journal, states = _selected_pair(
            other_binding,
            manifest_sha256=_manifest_sha(other_seed),
            status=other_status,
            prior_run_receipt=(
                _prior_triple(other_seed) if other_seed else None
            ),
            prior_collection_receipt=(
                _collection_triple(other_seed) if other_seed else None
            ),
        )
        store = StateStore(root / "state")
        with store.locked():
            store.write_aws_pair_transaction(
                _manifest_sha(other_seed),
                journal,
                states,
            )

        with pytest.raises(Exception) as caught:
            backend.submit(
                release=release,
                manifest=manifest,
                instance_id=manifest.instance_id,
                terminate_at=_fresh_deadline(),
                approval_path=root / "approval.json",
                apply=True,
                evidence=_selected_evidence(manifest),
                bootstrap_mode="bootstrap",
                prior_run_receipt=_prior_triple(1),
                prior_collection_receipt=_collection_triple(1),
            )

        assert getattr(caught.value, "code", None) == expected_code, case
        assert runner.calls == [], case
        with store.locked():
            assert store.read_aws_pair(manifest.sha256) is None
            for run in manifest.runs:
                assert store.read_run(run.run_id) is None


def test_selected_declared_and_resolved_bootstrap_must_match(
    tmp_path,
    monkeypatch,
):
    from msctl.errors import MsctlError
    from msctl.state import StateStore

    backend, lifecycle, runner = _selected_backend(
        tmp_path,
        monkeypatch,
        seed=0,
    )
    manifest = _selected_flow_manifest(lifecycle.binding)
    release = _selected_release(manifest)
    backend.approval_verifier = lambda **_kwargs: {}

    receipt_payload = _canonical(
        _controller_bootstrap_receipt(manifest, backend)
    )
    runner.outputs = [
        _download(
            receipt_payload,
            {
                "receipt": {
                    "content_length": len(receipt_payload),
                    "version_id": "bootstrap-receipt-1",
                }
            },
        ),
    ]
    with pytest.raises(Exception) as caught:
        backend.submit(
            release=release,
            manifest=manifest,
            instance_id=manifest.instance_id,
            terminate_at=_fresh_deadline(),
            approval_path=tmp_path / "approval.json",
            apply=True,
            evidence=_selected_evidence(manifest),
            bootstrap_mode="bootstrap",
            prior_run_receipt=None,
        )
    assert getattr(caught.value, "code", None) == "BOOTSTRAP_REUSE_INVALID"
    assert not runner.outputs
    assert not any(
        "create-tags" in argv or "send-command" in argv
        for argv, _operation in runner.calls
    )

    runner.calls.clear()
    runner.outputs = [
        MsctlError("AWS_COMMAND_FAILED", "no bootstrap receipt"),
    ]
    with pytest.raises(Exception) as caught:
        backend.submit(
            release=release,
            manifest=manifest,
            instance_id=manifest.instance_id,
            terminate_at=_fresh_deadline(),
            approval_path=tmp_path / "approval.json",
            apply=True,
            evidence=_selected_evidence(manifest),
            bootstrap_mode="reuse",
            prior_run_receipt=None,
        )
    assert getattr(caught.value, "code", None) == "BOOTSTRAP_REUSE_INVALID"
    store = StateStore(tmp_path / "state")
    with store.locked():
        assert store.read_aws_pair(manifest.sha256) is None


def test_selected_stale_boot_resolves_bootstrap_and_foreign_receipt_fails(
    tmp_path,
    monkeypatch,
):
    backend, lifecycle, runner = _selected_backend(
        tmp_path,
        monkeypatch,
        seed=0,
    )
    manifest = _selected_flow_manifest(lifecycle.binding)
    release = _selected_release(manifest)
    backend.approval_verifier = lambda **_kwargs: {}

    stale = _controller_bootstrap_receipt(manifest, backend)
    stale["boot_id"] = "87654321-4321-4cba-9abc-0987654321fe"
    stale_payload = _canonical(stale)
    runner.outputs = [
        _download(
            stale_payload,
            {
                "receipt": {
                    "content_length": len(stale_payload),
                    "version_id": "bootstrap-receipt-1",
                }
            },
        ),
        *_submit_tail_outputs(manifest, backend, bound_discovery=False),
    ]
    result = backend.submit(
        release=release,
        manifest=manifest,
        instance_id=manifest.instance_id,
        terminate_at=_fresh_deadline(),
        approval_path=tmp_path / "approval.json",
        apply=True,
        evidence=_selected_evidence(manifest),
        bootstrap_mode="bootstrap",
        prior_run_receipt=None,
    )
    assert result["submitted"] == 1

    foreign_root = tmp_path / "foreign"
    backend2, lifecycle2, runner2 = _selected_backend(
        foreign_root,
        monkeypatch,
        seed=0,
    )
    manifest2 = _selected_flow_manifest(lifecycle2.binding)
    release2 = _selected_release(manifest2)
    backend2.approval_verifier = lambda **_kwargs: {}
    foreign = _controller_bootstrap_receipt(manifest2, backend2)
    foreign["release_sha256"] = "a" * 64
    foreign["release_root"] = "releases/" + "a" * 64
    foreign_payload = _canonical(foreign)
    runner2.outputs = [
        _download(
            foreign_payload,
            {
                "receipt": {
                    "content_length": len(foreign_payload),
                    "version_id": "bootstrap-receipt-1",
                }
            },
        )
    ]
    with pytest.raises(Exception) as caught:
        backend2.submit(
            release=release2,
            manifest=manifest2,
            instance_id=manifest2.instance_id,
            terminate_at=_fresh_deadline(),
            approval_path=foreign_root / "approval.json",
            apply=True,
            evidence=_selected_evidence(manifest2),
            bootstrap_mode="bootstrap",
            prior_run_receipt=None,
        )
    assert getattr(caught.value, "code", None) == "BOOTSTRAP_REUSE_INVALID"


def _submitted_selected_pair(tmp_path, monkeypatch, *, seed: int, status: str):
    backend, lifecycle, runner = _selected_backend(
        tmp_path,
        monkeypatch,
        seed=seed,
    )
    manifest = _selected_flow_manifest(lifecycle.binding)
    release = _selected_release(manifest)
    evidence = _selected_evidence(manifest)
    terminate_at = _fresh_deadline()
    backend.approval_verifier = lambda **_kwargs: {}
    from msctl.errors import MsctlError

    runner.outputs = [
        MsctlError("AWS_COMMAND_FAILED", "no bootstrap receipt"),
        *_submit_tail_outputs(manifest, backend, bound_discovery=False),
    ]
    result = backend.submit(
        release=release,
        manifest=manifest,
        instance_id=manifest.instance_id,
        terminate_at=terminate_at,
        approval_path=tmp_path / "approval.json",
        apply=True,
        evidence=evidence,
        bootstrap_mode="bootstrap",
        prior_run_receipt=None,
    )
    assert result["submitted"] == 1
    from msctl.state import StateStore

    store = StateStore(tmp_path / "state")
    if status != "Pending":
        with store.locked():
            states = [
                store.read_run(run.run_id) for run in manifest.runs
            ]
            refreshed = backend._refresh_paired_states(
                store,
                manifest,
                states,
                {"status": status},
            )
            del refreshed
    runner.calls.clear()
    runner.outputs = []
    return SimpleNamespace(
        backend=backend,
        evidence=evidence,
        manifest=manifest,
        release=release,
        runner=runner,
        store=store,
        submit_terminate_at=terminate_at,
    )


def _selected_checkpoint_receipt(manifest):
    receipt_sha256 = "a" * 64
    checkpoints = []
    for run in manifest.runs:
        digest = ("1" if run.arm == "dense" else "2") * 64
        checkpoints.append(
            SimpleNamespace(
                arm=run.arm,
                run_id=run.run_id,
                seed=manifest.seed,
                world_size=4,
                step=1_358,
                config_sha256=run.config_sha256,
                config_fingerprint="3" * 64,
                checkpoint_version=1,
                data=SimpleNamespace(
                    build_id=manifest.dataset_build_id,
                    receipt_sha256=manifest.dataset_receipt_sha256,
                    global_cursor=1,
                    ordered_stream_sha256=(
                        manifest.ordered_stream_sha256
                    ),
                    sidecar_name="sidecar.json",
                ),
                object=SimpleNamespace(
                    bytes=100,
                    sha256=digest,
                    uri=(
                        f"{S3_ROOT}/checkpoints/seed-{manifest.seed}/"
                        f"{run.arm}/sha256/{digest}.pt"
                    ),
                    version_id=f"{run.arm}-version-1",
                ),
            )
        )
    return SimpleNamespace(
        schema_version=3,
        sha256=receipt_sha256,
        bytes=1_000,
        uri=(
            f"{S3_ROOT}/receipts/checkpoints/seed-{manifest.seed}/"
            f"sha256/{receipt_sha256}.json"
        ),
        version_id="receipt-version-1",
        checkpoints=tuple(checkpoints),
        provider=manifest.provider,
        profile_id=manifest.profile_id,
        profile_sha256=manifest.profile_sha256,
        hardware_amendment_sha256=manifest.hardware_amendment_sha256,
        provider_selection_sha256=manifest.provider_selection_sha256,
        provider_selection_version_id=(
            manifest.provider_selection_version_id
        ),
        runtime_lock_sha256=manifest.runtime_lock_sha256,
        runtime_sbom_sha256=manifest.runtime_sbom_sha256,
        qualification_evidence_sha256=(
            manifest.qualification_evidence_sha256
        ),
        environment_receipt_sha256="8" * 64,
        qualification_canary_receipt_sha256=(
            manifest.qualification_canary_receipt_sha256
        ),
        qualification_approval_receipt_sha256=(
            manifest.qualification_approval_receipt_sha256
        ),
        objective_controls_contract_sha256=(
            manifest.objective_controls_contract_sha256
        ),
    )


def test_selected_resume_flows_fresh_deadline_through_every_binding(
    tmp_path,
    monkeypatch,
):
    from msctl.aws_argv import _validate_intent
    from msctl.jsonutil import canonical_sha256

    import msctl.aws_p5 as aws_p5_module

    fixture = _submitted_selected_pair(
        tmp_path,
        monkeypatch,
        seed=0,
        status="Failed",
    )
    backend = fixture.backend
    manifest = fixture.manifest
    monkeypatch.setattr(
        backend,
        "_checkpoint_map",
        lambda _manifest, receipt: {
            checkpoint.arm: checkpoint
            for checkpoint in receipt.checkpoints
        },
    )
    monkeypatch.setattr(
        aws_p5_module,
        "verify_aws_checkpoint_receipt_v3",
        lambda *_args, **_kwargs: None,
    )
    receipt = _selected_checkpoint_receipt(manifest)
    fresh_deadline = (
        (datetime.now(UTC) + timedelta(hours=9))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    assert fresh_deadline != fixture.submit_terminate_at
    approval_record: dict[str, object] = {}
    backend.approval_verifier = _approval_recorder(
        approval_record,
        fixture.runner,
    )

    receipt_payload = _canonical(
        _controller_bootstrap_receipt(manifest, backend)
    )
    receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()

    def _put(argv: list[str]):
        checksum = argv[argv.index("--checksum-sha256") + 1]
        return {
            "object": {
                "checksum_sha256": checksum,
                "version_id": "intent-version-2",
            }
        }

    def _head(argv: list[str]):
        del argv
        intent_path = sorted(
            Path(backend.state_root).glob("intent-*.json"),
            key=lambda path: path.stat().st_mtime,
        )[-1]
        payload = intent_path.read_bytes()
        return {
            "object": {
                "checksum_sha256": __import__("base64").b64encode(
                    hashlib.sha256(payload).digest()
                ).decode("ascii"),
                "content_length": len(payload),
                "metadata": {
                    "operation-id": json.loads(payload)["operation_id"],
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                "version_id": "intent-version-2",
            }
        }

    fixture.runner.outputs = [
        _download(
            receipt_payload,
            {
                "receipt": {
                    "content_length": len(receipt_payload),
                    "version_id": "bootstrap-receipt-1",
                }
            },
        ),
        {
            "instances": [
                _selected_identity_instance(manifest, backend, bound=True)
            ]
        },
        {
            "attribute": {
                "instance_id": manifest.instance_id,
                "shutdown_behavior": "terminate",
            }
        },
        {
            "command": {
                "command_id": "cmd-0123456789abcdef0",
                "status": "Failed",
            }
        },
        {
            "managed_instances": [
                {
                    "instance_id": manifest.instance_id,
                    "ping_status": "Online",
                }
            ]
        },
        _argv_document_listing(),
        _put,
        _head,
        {"command": {"command_id": "cmd-0123456789abcdef1"}},
    ]

    result = backend.resume(
        release=fixture.release,
        manifest=manifest,
        checkpoint_receipt=receipt,
        approval_path=tmp_path / "approval.json",
        apply=True,
        evidence=fixture.evidence,
        terminate_at=fresh_deadline,
        bootstrap_mode="reuse",
    )

    assert result["submitted"] == 1
    assert result["attempt"] == 2
    resources = approval_record["resources"]
    assert resources["terminate_at"] == fresh_deadline
    assert resources["bootstrap_mode"] == "reuse"
    assert approval_record["calls_at_approval"] == 0

    intent_path = sorted(
        Path(backend.state_root).glob("intent-*.json"),
        key=lambda path: path.stat().st_mtime,
    )[-1]
    payload = intent_path.read_bytes()
    intent = _validate_intent(
        payload,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert intent["operation"] == "resume"
    assert intent["terminate_at"] == fresh_deadline
    assert intent["bootstrap"] == {
        "mode": "reuse",
        "receipt_sha256": receipt_sha256,
    }
    assert intent["lease_unit"] == (
        "memorysplit-auto-terminate-"
        + canonical_sha256(
            {
                "attempt": 2,
                "operation": "resume",
                "run_manifest_sha256": manifest.sha256,
                "seed": manifest.seed,
                "terminate_at": fresh_deadline,
            }
        )
    )
    names = [step["name"] for step in intent["steps"]]
    assert names == [
        "reset-termination-leases",
        "auto-termination",
        "verify-bootstrap-reuse",
        "prepare-resume-staging",
        "materialize-resume-receipt",
        "materialize-resume-dense",
        "materialize-resume-split90",
        "paired-launch",
    ]

    with fixture.store.locked():
        stored = fixture.store.read_run(str(manifest.runs[0].run_id))
    assert stored["terminate_at"] == fresh_deadline
    assert stored["bootstrap_mode"] == "reuse"
    assert stored["bootstrap_receipt_sha256"] == receipt_sha256
    assert stored["prior_run_receipt"] is None
    assert stored["prior_collection_receipt"] is None
    assert stored["attempt"] == 2


def test_selected_resume_requires_fresh_deadline_and_mode(
    tmp_path,
    monkeypatch,
):
    fixture = _submitted_selected_pair(
        tmp_path,
        monkeypatch,
        seed=0,
        status="Failed",
    )
    receipt = _selected_checkpoint_receipt(fixture.manifest)
    monkeypatch.setattr(
        fixture.backend,
        "_checkpoint_map",
        lambda _manifest, value: {
            checkpoint.arm: checkpoint
            for checkpoint in value.checkpoints
        },
    )

    with pytest.raises(Exception) as caught:
        fixture.backend.resume(
            release=fixture.release,
            manifest=fixture.manifest,
            checkpoint_receipt=receipt,
            approval_path=tmp_path / "approval.json",
            apply=True,
            evidence=fixture.evidence,
            terminate_at=None,
            bootstrap_mode="reuse",
        )
    assert getattr(caught.value, "code", None) == (
        "TERMINATION_DEADLINE_INVALID"
    )
    assert fixture.runner.calls == []

    with pytest.raises(Exception) as caught:
        fixture.backend.resume(
            release=fixture.release,
            manifest=fixture.manifest,
            checkpoint_receipt=receipt,
            approval_path=tmp_path / "approval.json",
            apply=True,
            evidence=fixture.evidence,
            terminate_at=_fresh_deadline(),
            bootstrap_mode=None,
        )
    assert getattr(caught.value, "code", None) == "CLI_USAGE"
    assert fixture.runner.calls == []


def test_selected_intent_shapes_validate_against_remote_wrapper(
    tmp_path,
    monkeypatch,
):
    from msctl.aws_argv import _validate_intent
    from msctl.jsonutil import canonical_json

    backend, lifecycle, runner = _selected_backend(
        tmp_path,
        monkeypatch,
        seed=4,
    )
    del runner
    manifest = _selected_flow_manifest(lifecycle.binding)
    release = _selected_release(manifest)
    evidence = _selected_evidence(manifest)
    terminate_at = _fresh_deadline()
    monkeypatch.setattr(
        backend,
        "_checkpoint_map",
        lambda _manifest, receipt: {
            checkpoint.arm: checkpoint
            for checkpoint in receipt.checkpoints
        },
    )
    receipt = _selected_checkpoint_receipt(manifest)

    for operation, bootstrap_mode, attempt in (
        ("submit", "bootstrap", 1),
        ("submit", "reuse", 1),
        ("resume", "reuse", 2),
        ("resume", "bootstrap", 2),
    ):
        checkpoints = (
            {
                checkpoint.arm: checkpoint
                for checkpoint in receipt.checkpoints
            }
            if operation == "resume"
            else None
        )
        core = backend._training_operation_intent(
            operation=operation,
            release=release,
            manifest=manifest,
            terminate_at=terminate_at,
            evidence=evidence,
            attempt=attempt,
            bootstrap_mode=bootstrap_mode,
            bootstrap_receipt_sha256=(
                "b" * 64 if bootstrap_mode == "reuse" else None
            ),
            checkpoints=checkpoints,
            checkpoint_receipt_sha256=(
                receipt.sha256 if operation == "resume" else None
            ),
            checkpoint_receipt_uri=(
                receipt.uri if operation == "resume" else None
            ),
            checkpoint_receipt_version_id=(
                receipt.version_id if operation == "resume" else None
            ),
            checkpoint_receipt_bytes=(
                receipt.bytes if operation == "resume" else None
            ),
        )
        envelope = backend._operation_envelope(
            core,
            instance_id=manifest.instance_id,
            terminate_at=terminate_at,
        )
        payload = canonical_json(envelope)
        parsed = _validate_intent(
            payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
        assert parsed["schema_version"] == 3
        assert parsed["bootstrap"]["mode"] == bootstrap_mode
        assert not any(
            step["name"]
            in {
                "prepare-staging",
                "materialize-release-archive",
                "materialize-release-receipt",
                "materialize-cohort-assignment",
                "materialize-dataset",
            }
            for step in parsed["steps"]
        )


def test_legacy_tag_binding_is_byte_identical(tmp_path):
    from msctl.aws_p5 import AwsP5Backend
    from tests.test_msctl import (
        _FakeAwsRunner,
        _aws_manifest_object,
        _aws_profile_object,
        _aws_runtime_object,
    )

    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=tmp_path / "state",
        runner=_FakeAwsRunner(),
    )
    manifest = _aws_manifest_object()
    tags = json.loads(
        backend._instance_tags(
            manifest,
            terminate_at="2099-01-01T00:00:00Z",
        )
    )
    assert [tag["Key"] for tag in tags] == [
        "MemorySplitProvider",
        "MemorySplitSeed",
        "MemorySplitCohortSHA256",
        "MemorySplitReleaseSHA256",
        "MemorySplitDatasetSHA256",
        "MemorySplitRunManifestSHA256",
        "MemorySplitProfileSHA256",
        "MemorySplitRuntimeSHA256",
        "MemorySplitContainerDigest",
        "MemorySplitTerminateAt",
    ]


def test_selected_identity_tags_are_seed_independent(tmp_path, monkeypatch):
    backend0, lifecycle0, _runner0 = _selected_backend(
        tmp_path / "seed0",
        monkeypatch,
        seed=0,
    )
    backend1, lifecycle1, _runner1 = _selected_backend(
        tmp_path / "seed1",
        monkeypatch,
        seed=1,
    )
    tags0 = backend0._selected_identity_tags(
        _selected_flow_manifest(lifecycle0.binding)
    )
    tags1 = backend1._selected_identity_tags(
        _selected_flow_manifest(lifecycle1.binding)
    )
    assert tags0 == tags1


def test_prior_evidence_head_drift_blocks_before_any_binding(
    tmp_path,
    monkeypatch,
):
    import base64

    backend, lifecycle, runner = _selected_backend(
        tmp_path,
        monkeypatch,
        seed=1,
    )
    manifest = _selected_flow_manifest(lifecycle.binding)
    release = _selected_release(manifest)
    backend.approval_verifier = lambda **_kwargs: {}
    prior_binding = provider_lifecycle(
        load_aws_gpu_profile(P5_PROFILE),
        seed=1,
    ).binding
    prior_value = _receipt_value(prior_binding, seed=0)
    prior_payload, prior_ref = _receipt_ref(prior_value)

    runner.outputs = [
        _download(
            prior_payload,
            {
                "receipt": {
                    "checksum_sha256": base64.b64encode(
                        bytes.fromhex(prior_ref.sha256)
                    ).decode("ascii"),
                    "version_id": prior_ref.version_id,
                }
            },
        ),
        {
            "object": {
                "checksum_sha256": base64.b64encode(
                    bytes.fromhex("f" * 64)
                ).decode("ascii"),
                "content_length": 1,
                "version_id": "drifted",
            }
        },
    ]
    with pytest.raises(Exception) as caught:
        backend.submit(
            release=release,
            manifest=manifest,
            instance_id=manifest.instance_id,
            terminate_at=_fresh_deadline(),
            approval_path=tmp_path / "approval.json",
            apply=True,
            evidence=_selected_evidence(manifest),
            bootstrap_mode="bootstrap",
            prior_run_receipt={
                "uri": prior_ref.uri,
                "sha256": prior_ref.sha256,
                "version_id": prior_ref.version_id,
            },
            prior_collection_receipt=_collection_triple(1),
        )
    assert getattr(caught.value, "code", None) == "SEED_TRANSITION_BLOCKED"
    assert len(runner.calls) == 2
    from msctl.state import StateStore

    store = StateStore(tmp_path / "state")
    with store.locked():
        assert store.read_aws_pair(manifest.sha256) is None


def test_prior_receipt_dataclasses_fail_closed():
    from msctl.aws_seed_transition import (
        AdmittedPriorRun,
        PriorEvidenceObject,
        PriorRunReceiptRef,
        SeedTransitionError,
    )

    with pytest.raises(SeedTransitionError):
        PriorRunReceiptRef(uri="https://x/y", sha256="a" * 64, version_id="v")
    with pytest.raises(SeedTransitionError):
        PriorRunReceiptRef(
            uri="s3://bucket/key.json",
            sha256="A" * 64,
            version_id="v",
        )
    with pytest.raises(SeedTransitionError):
        PriorRunReceiptRef(
            uri="s3://bucket/key.json",
            sha256="a" * 64,
            version_id="null",
        )
    with pytest.raises(SeedTransitionError):
        PriorEvidenceObject(
            uri="s3://bucket/object.pt",
            sha256="a" * 64,
            bytes=0,
            version_id="v",
        )
    with pytest.raises(SeedTransitionError):
        PriorEvidenceObject(
            uri="s3://bucket/object.pt",
            sha256="a" * 64,
            bytes=True,
            version_id="v",
        )
    ref = PriorRunReceiptRef(
        uri="s3://bucket/key.json",
        sha256="a" * 64,
        version_id="v1",
    )
    evidence = PriorEvidenceObject(
        uri="s3://bucket/object.pt",
        sha256="a" * 64,
        bytes=None,
        version_id="v1",
    )
    with pytest.raises(SeedTransitionError, match="14"):
        AdmittedPriorRun(seed=0, receipt=ref, evidence=(evidence,) * 13)
    with pytest.raises(SeedTransitionError, match="seed"):
        AdmittedPriorRun(seed=9, receipt=ref, evidence=(evidence,) * 14)
    admitted = AdmittedPriorRun(
        seed=8,
        receipt=ref,
        evidence=(evidence,) * 14,
    )
    assert admitted.seed == 8


def test_selected_submit_requires_both_prior_triples(tmp_path, monkeypatch):
    from msctl.state import StateStore

    for case, run_triple, collection_triple in (
        ("missing-collection", _prior_triple(1), None),
        ("missing-run", None, _collection_triple(1)),
        ("missing-both", None, None),
    ):
        root = tmp_path / case
        backend, lifecycle, runner = _selected_backend(
            root,
            monkeypatch,
            seed=1,
        )
        manifest = _selected_flow_manifest(lifecycle.binding)
        release = _selected_release(manifest)
        backend.approval_verifier = lambda **_kwargs: {}

        with pytest.raises(Exception) as caught:
            backend.submit(
                release=release,
                manifest=manifest,
                instance_id=manifest.instance_id,
                terminate_at=_fresh_deadline(),
                approval_path=root / "approval.json",
                apply=True,
                evidence=_selected_evidence(manifest),
                bootstrap_mode="bootstrap",
                prior_run_receipt=run_triple,
                prior_collection_receipt=collection_triple,
            )

        assert getattr(caught.value, "code", None) == (
            "SEED_TRANSITION_BLOCKED"
        ), case
        assert runner.calls == [], case
        store = StateStore(root / "state")
        with store.locked():
            assert store.read_aws_pair(manifest.sha256) is None


def test_prior_collection_drift_blocks_before_any_binding(
    tmp_path,
    monkeypatch,
):
    from msctl.aws_seed_transition import admit_prior_seed_finalization
    from msctl.state import StateStore

    for case in ("foreign-collection", "checkpoint-head-drift"):
        root = tmp_path / case
        backend, lifecycle, runner = _selected_backend(
            root,
            monkeypatch,
            seed=1,
        )
        manifest = _selected_flow_manifest(lifecycle.binding)
        release = _selected_release(manifest)
        backend.approval_verifier = lambda **_kwargs: {}
        prior_binding = provider_lifecycle(
            load_aws_gpu_profile(P5_PROFILE),
            seed=1,
        ).binding
        prior_value = _receipt_value(prior_binding, seed=0)
        prior_payload, prior_ref = _receipt_ref(prior_value)
        admitted = admit_prior_seed_finalization(
            prior_payload,
            ref=prior_ref,
            **_admit_arguments(prior_binding),
        )
        if case == "foreign-collection":
            # A collection receipt for a different finalization of the same
            # seed parses cleanly but must not admit this transition.
            foreign_value = copy.deepcopy(prior_value)
            foreign_value["request_id"] = "7" * 32
            foreign_payload, foreign_ref = _receipt_ref(foreign_value)
            collection_value, collection_payload, collection_ref = (
                _prior_collection_bits(
                    foreign_value,
                    foreign_payload,
                    foreign_ref,
                )
            )
            expected_calls = 16
            outputs = [
                _prior_receipt_download(prior_payload, prior_ref),
                *_evidence_head_outputs(admitted),
                _prior_receipt_download(
                    collection_payload,
                    collection_ref,
                ),
            ]
        else:
            collection_value, collection_payload, collection_ref = (
                _prior_collection_bits(
                    prior_value,
                    prior_payload,
                    prior_ref,
                )
            )
            heads = _collection_checkpoint_head_outputs(collection_value)
            heads[0]["object"]["version_id"] = "drifted-version"
            expected_calls = 17
            outputs = [
                _prior_receipt_download(prior_payload, prior_ref),
                *_evidence_head_outputs(admitted),
                _prior_receipt_download(
                    collection_payload,
                    collection_ref,
                ),
                heads[0],
            ]
        runner.outputs = outputs

        with pytest.raises(Exception) as caught:
            backend.submit(
                release=release,
                manifest=manifest,
                instance_id=manifest.instance_id,
                terminate_at=_fresh_deadline(),
                approval_path=root / "approval.json",
                apply=True,
                evidence=_selected_evidence(manifest),
                bootstrap_mode="bootstrap",
                prior_run_receipt={
                    "uri": prior_ref.uri,
                    "sha256": prior_ref.sha256,
                    "version_id": prior_ref.version_id,
                },
                prior_collection_receipt={
                    "uri": collection_ref.uri,
                    "sha256": collection_ref.sha256,
                    "version_id": collection_ref.version_id,
                },
            )

        assert getattr(caught.value, "code", None) == (
            "SEED_TRANSITION_BLOCKED"
        ), case
        assert len(runner.calls) == expected_calls, case
        assert not any(
            "create-tags" in argv or "send-command" in argv
            for argv, _operation in runner.calls
        ), case
        store = StateStore(root / "state")
        with store.locked():
            assert store.read_aws_pair(manifest.sha256) is None


def test_selected_submit_replay_compares_stored_collection_triple(
    tmp_path,
    monkeypatch,
):
    from msctl.aws_seed_transition import admit_prior_seed_finalization
    from msctl.errors import MsctlError

    backend, lifecycle, runner = _selected_backend(
        tmp_path,
        monkeypatch,
        seed=1,
    )
    manifest = _selected_flow_manifest(lifecycle.binding)
    release = _selected_release(manifest)
    evidence = _selected_evidence(manifest)
    terminate_at = _fresh_deadline()
    backend.approval_verifier = lambda **_kwargs: {}

    prior_binding = provider_lifecycle(
        load_aws_gpu_profile(P5_PROFILE),
        seed=1,
    ).binding
    prior_value = _receipt_value(prior_binding, seed=0)
    prior_payload, prior_ref = _receipt_ref(prior_value)
    admitted = admit_prior_seed_finalization(
        prior_payload,
        ref=prior_ref,
        **_admit_arguments(prior_binding),
    )
    collection_value, collection_payload, collection_ref = (
        _prior_collection_bits(prior_value, prior_payload, prior_ref)
    )
    run_triple = {
        "uri": prior_ref.uri,
        "sha256": prior_ref.sha256,
        "version_id": prior_ref.version_id,
    }
    collection_triple = {
        "uri": collection_ref.uri,
        "sha256": collection_ref.sha256,
        "version_id": collection_ref.version_id,
    }
    runner.outputs = [
        _prior_receipt_download(prior_payload, prior_ref),
        *_evidence_head_outputs(admitted),
        _prior_receipt_download(collection_payload, collection_ref),
        *_collection_checkpoint_head_outputs(collection_value),
        MsctlError("AWS_COMMAND_FAILED", "no bootstrap receipt"),
        *_submit_tail_outputs(manifest, backend, bound_discovery=False),
    ]
    submit_arguments = {
        "release": release,
        "manifest": manifest,
        "instance_id": manifest.instance_id,
        "terminate_at": terminate_at,
        "approval_path": tmp_path / "approval.json",
        "apply": True,
        "evidence": evidence,
        "bootstrap_mode": "bootstrap",
        "prior_run_receipt": run_triple,
        "prior_collection_receipt": collection_triple,
    }
    assert backend.submit(**submit_arguments)["submitted"] == 1

    runner.calls.clear()
    drifted = dict(submit_arguments)
    drifted["prior_collection_receipt"] = _collection_triple(1)
    with pytest.raises(Exception) as caught:
        backend.submit(**drifted)
    assert getattr(caught.value, "code", None) == "BOOTSTRAP_REUSE_INVALID"
    assert runner.calls == []

    runner.calls.clear()
    runner.outputs = [
        {
            "command": {
                "command_id": "cmd-0123456789abcdef0",
                "status": "InProgress",
            }
        },
    ]
    replay = backend.submit(**submit_arguments)
    assert replay["idempotent"] is True
    assert not any(
        "get-object" in argv or "head-object" in argv
        for argv, _operation in runner.calls
    )


def test_selected_resume_readmits_stored_collection_before_bootstrap(
    tmp_path,
    monkeypatch,
):
    from msctl.aws_seed_transition import admit_prior_seed_finalization
    from msctl.errors import MsctlError

    import msctl.aws_p5 as aws_p5_module

    backend, lifecycle, runner = _selected_backend(
        tmp_path,
        monkeypatch,
        seed=1,
    )
    manifest = _selected_flow_manifest(lifecycle.binding)
    release = _selected_release(manifest)
    evidence = _selected_evidence(manifest)
    backend.approval_verifier = lambda **_kwargs: {}
    prior_binding = provider_lifecycle(
        load_aws_gpu_profile(P5_PROFILE),
        seed=1,
    ).binding
    prior_value = _receipt_value(prior_binding, seed=0)
    prior_payload, prior_ref = _receipt_ref(prior_value)
    admitted = admit_prior_seed_finalization(
        prior_payload,
        ref=prior_ref,
        **_admit_arguments(prior_binding),
    )
    collection_value, collection_payload, collection_ref = (
        _prior_collection_bits(prior_value, prior_payload, prior_ref)
    )
    runner.outputs = [
        _prior_receipt_download(prior_payload, prior_ref),
        *_evidence_head_outputs(admitted),
        _prior_receipt_download(collection_payload, collection_ref),
        *_collection_checkpoint_head_outputs(collection_value),
        MsctlError("AWS_COMMAND_FAILED", "no bootstrap receipt"),
        *_submit_tail_outputs(manifest, backend, bound_discovery=False),
    ]
    submitted = backend.submit(
        release=release,
        manifest=manifest,
        instance_id=manifest.instance_id,
        terminate_at=_fresh_deadline(),
        approval_path=tmp_path / "approval.json",
        apply=True,
        evidence=evidence,
        bootstrap_mode="bootstrap",
        prior_run_receipt={
            "uri": prior_ref.uri,
            "sha256": prior_ref.sha256,
            "version_id": prior_ref.version_id,
        },
        prior_collection_receipt={
            "uri": collection_ref.uri,
            "sha256": collection_ref.sha256,
            "version_id": collection_ref.version_id,
        },
    )
    assert submitted["submitted"] == 1
    from msctl.state import StateStore

    store = StateStore(tmp_path / "state")
    with store.locked():
        states = [store.read_run(run.run_id) for run in manifest.runs]
        backend._refresh_paired_states(
            store,
            manifest,
            states,
            {"status": "Failed"},
        )

    monkeypatch.setattr(
        backend,
        "_checkpoint_map",
        lambda _manifest, receipt: {
            checkpoint.arm: checkpoint
            for checkpoint in receipt.checkpoints
        },
    )
    monkeypatch.setattr(
        aws_p5_module,
        "verify_aws_checkpoint_receipt_v3",
        lambda *_args, **_kwargs: None,
    )
    receipt = _selected_checkpoint_receipt(manifest)
    bootstrap_payload = _canonical(
        _controller_bootstrap_receipt(manifest, backend)
    )

    def _put(argv: list[str]):
        checksum = argv[argv.index("--checksum-sha256") + 1]
        return {
            "object": {
                "checksum_sha256": checksum,
                "version_id": "intent-version-2",
            }
        }

    def _head(argv: list[str]):
        del argv
        intent_path = sorted(
            Path(backend.state_root).glob("intent-*.json"),
            key=lambda path: path.stat().st_mtime,
        )[-1]
        payload = intent_path.read_bytes()
        return {
            "object": {
                "checksum_sha256": __import__("base64").b64encode(
                    hashlib.sha256(payload).digest()
                ).decode("ascii"),
                "content_length": len(payload),
                "metadata": {
                    "operation-id": json.loads(payload)["operation_id"],
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                "version_id": "intent-version-2",
            }
        }

    runner.calls.clear()
    runner.outputs = [
        _prior_receipt_download(prior_payload, prior_ref),
        *_evidence_head_outputs(admitted),
        _prior_receipt_download(collection_payload, collection_ref),
        *_collection_checkpoint_head_outputs(collection_value),
        _download(
            bootstrap_payload,
            {
                "receipt": {
                    "content_length": len(bootstrap_payload),
                    "version_id": "bootstrap-receipt-1",
                }
            },
        ),
        {
            "instances": [
                _selected_identity_instance(manifest, backend, bound=True)
            ]
        },
        {
            "attribute": {
                "instance_id": manifest.instance_id,
                "shutdown_behavior": "terminate",
            }
        },
        {
            "command": {
                "command_id": "cmd-0123456789abcdef0",
                "status": "Failed",
            }
        },
        {
            "managed_instances": [
                {
                    "instance_id": manifest.instance_id,
                    "ping_status": "Online",
                }
            ]
        },
        _argv_document_listing(),
        _put,
        _head,
        {"command": {"command_id": "cmd-0123456789abcdef1"}},
    ]

    result = backend.resume(
        release=release,
        manifest=manifest,
        checkpoint_receipt=receipt,
        approval_path=tmp_path / "approval.json",
        apply=True,
        evidence=evidence,
        terminate_at=_fresh_deadline(),
        bootstrap_mode="reuse",
    )

    assert result["submitted"] == 1
    operations = [operation for _argv, operation in runner.calls]
    assert operations.count("verify prior collection checkpoint") == 2
    assert operations.index("fetch prior seed collection") < (
        operations.index("resolve bootstrap receipt")
    )
    with store.locked():
        stored = store.read_run(str(manifest.runs[0].run_id))
    assert stored["prior_collection_receipt"] == {
        "uri": collection_ref.uri,
        "sha256": collection_ref.sha256,
        "version_id": collection_ref.version_id,
    }


def test_controller_bootstrap_receipt_fields_mirror_bootstrap_contract():
    from cluster.aws.p5.bootstrap import BOOTSTRAP_RECEIPT_FIELDS
    from msctl.aws_p5 import _BOOTSTRAP_RECEIPT_FIELDS

    assert tuple(sorted(_BOOTSTRAP_RECEIPT_FIELDS)) == tuple(
        sorted(BOOTSTRAP_RECEIPT_FIELDS)
    )
    assert len(_BOOTSTRAP_RECEIPT_FIELDS) == len(
        set(_BOOTSTRAP_RECEIPT_FIELDS)
    )
