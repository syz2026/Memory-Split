from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cluster.aws.gpu_profile import load_aws_gpu_profile
from msctl.aws_contracts import (
    ARMS,
    COHORT_ID,
    SEEDS,
    SNAPSHOT_STEPS,
    EVALUATION_OUTPUT_MEMBERS,
    cohort_collection_receipt_key,
    cohort_report_object_key,
    collection_receipt_key,
    evaluation_output_member_key,
    study_lock_object_key,
)
from msctl.aws_cohort_collect import (
    COHORT_COLLECTION_OBJECT_COUNT,
    COHORT_COLLECTION_RECEIPT_FIELDS,
    COHORT_COLLECTION_RECEIPT_TYPE,
    CohortCollectedObject,
    CohortCollectionError,
    CohortEvidenceRef,
    parse_cohort_collection_receipt_bytes,
    parse_cohort_evidence_index,
)
from tests.study_lock_fixtures import (
    BOOT_ID,
    INSTANCE_ID,
    RELEASE_RECEIPT_SHA256,
    RELEASE_SHA256,
    S3_ROOT,
    SOURCE_COMMIT,
    SOURCE_TREE,
    canonical_receipt_bytes,
    default_provider_selection,
    digest,
    expected_lifecycle_binding,
    lifecycle_core,
)
from tests.test_aws_collect import _ScriptedDownloadRunner
from tests.test_aws_seed_transition import (
    IMAGE,
    IMAGE_DIGEST,
    P5_PROFILE,
    ROOT,
    _ScriptedRunner,
)


PROFILE_ID = "aws-p5.48xlarge-v3"
CONTROLLER_BOOT_ID = "87654321-4321-4cba-9abc-0987654321fe"
FOREIGN_INSTANCE_ID = "i-0fedcba9876543210"
_SLOTS = tuple(
    (seed, arm, step)
    for seed in SEEDS
    for arm in ARMS
    for step in SNAPSHOT_STEPS
)
_MEMBER_CEILING = 2 * 1024 * 1024 * 1024


def _output_id(seed: int, arm: str, step: int) -> str:
    return (
        f"snapshot-evaluation-memorysplit-v3-{PROFILE_ID}"
        f"-s{seed:02d}-{arm}-step{step:05d}"
    )


def _checksum(sha256: str) -> str:
    return base64.b64encode(bytes.fromhex(sha256)).decode("ascii")


# ---------------------------------------------------------------------------
# Synthetic (payload-free) builders for the parser matrices.
# ---------------------------------------------------------------------------


def _synthetic_lock_ref() -> dict[str, object]:
    lock_sha256 = digest("synthetic-study-lock")
    return {
        "uri": f"{S3_ROOT}/" + study_lock_object_key(lock_sha256),
        "sha256": lock_sha256,
        "version_id": "study-lock-version-1",
        "bytes": 8_192,
    }


def _synthetic_report_ref() -> dict[str, object]:
    report_sha256 = digest("synthetic-cohort-report")
    return {
        "uri": f"{S3_ROOT}/" + cohort_report_object_key(report_sha256),
        "sha256": report_sha256,
        "version_id": "cohort-report-version-1",
        "bytes": 16_384,
    }


def _synthetic_seed_rows() -> list[dict[str, object]]:
    rows = []
    for seed in SEEDS:
        sha256 = digest(f"synthetic-seed-collection:{seed}")
        rows.append(
            {
                "seed": seed,
                "uri": f"{S3_ROOT}/" + collection_receipt_key(seed, sha256),
                "sha256": sha256,
                "version_id": f"collection-version-{seed}",
                "bytes": 4_096 + seed,
            }
        )
    return rows


def _synthetic_member_rows(
    seed: int,
    arm: str,
    step: int,
) -> list[dict[str, object]]:
    rows = []
    for member in EVALUATION_OUTPUT_MEMBERS:
        sha256 = digest(f"synthetic-member:{seed}:{arm}:{step}:{member}")
        rows.append(
            {
                "member": member,
                "uri": (
                    f"{S3_ROOT}/"
                    + evaluation_output_member_key(
                        seed,
                        arm,
                        step,
                        member,
                        sha256,
                    )
                ),
                "sha256": sha256,
                "version_id": f"member-{seed}-{arm}-{step}-{member}",
                "bytes": 1_024 + step + len(member),
            }
        )
    return rows


def _synthetic_index() -> dict[str, object]:
    return {
        "schema_version": 1,
        "cohort_id": COHORT_ID,
        "study_lock": _synthetic_lock_ref(),
        "cohort_report": _synthetic_report_ref(),
        "seed_collections": _synthetic_seed_rows(),
        "outputs": [
            {
                "slot_index": index,
                "seed": seed,
                "arm": arm,
                "optimizer_step": step,
                "output_id": _output_id(seed, arm, step),
                "members": _synthetic_member_rows(seed, arm, step),
            }
            for index, (seed, arm, step) in enumerate(_SLOTS)
        ],
    }


def _synthetic_receipt_value(binding) -> dict[str, object]:
    lock_ref = _synthetic_lock_ref()
    report_ref = _synthetic_report_ref()
    seed_rows = _synthetic_seed_rows()
    objects: list[dict[str, object]] = []
    for seed, arm, step in _SLOTS:
        for member_row in _synthetic_member_rows(seed, arm, step):
            objects.append(
                {
                    "kind": "output_member",
                    "seed": seed,
                    "arm": arm,
                    "step": step,
                    "member": member_row["member"],
                    "uri": member_row["uri"],
                    "sha256": member_row["sha256"],
                    "bytes": member_row["bytes"],
                    "version_id": member_row["version_id"],
                }
            )
    for row in seed_rows:
        objects.append(
            {
                "kind": "seed_collection",
                "seed": row["seed"],
                "arm": None,
                "step": None,
                "member": None,
                "uri": row["uri"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "version_id": row["version_id"],
            }
        )
    for kind, reference in (
        ("study_lock", lock_ref),
        ("cohort_report", report_ref),
    ):
        objects.append(
            {
                "kind": kind,
                "seed": None,
                "arm": None,
                "step": None,
                "member": None,
                "uri": reference["uri"],
                "sha256": reference["sha256"],
                "bytes": reference["bytes"],
                "version_id": reference["version_id"],
            }
        )
    return {
        "boot_id": binding.boot_id,
        "canary_receipt_sha256": (
            binding.qualification_canary_receipt_sha256
        ),
        "cohort_id": binding.cohort_id,
        "cohort_report": report_ref,
        "collected_at": "2026-07-24T04:00:00Z",
        "complete": True,
        "dataset_build_id": digest("data-build"),
        "dataset_receipt_sha256": digest("data-receipt"),
        "environment_receipt_sha256": (
            binding.qualification_environment_receipt_sha256
        ),
        "hardware_amendment_sha256": binding.hardware_amendment_sha256,
        "instance_id": binding.instance_id,
        "objective_controls_contract_sha256": (
            binding.objective_controls_contract_sha256
        ),
        "objects": objects,
        "ordered_stream_sha256": digest("ordered-stream"),
        "preregistration_sha256": digest("preregistration"),
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
        "receipt_type": COHORT_COLLECTION_RECEIPT_TYPE,
        "release_receipt_sha256": RELEASE_RECEIPT_SHA256,
        "release_sha256": RELEASE_SHA256,
        "request_id": "b" * 32,
        "run_manifest_sha256": digest("run-manifest-9"),
        "runtime_lock_sha256": binding.runtime_lock_sha256,
        "runtime_sbom_sha256": binding.runtime_sbom_sha256,
        "schema_version": 3,
        "sealed_evaluation_release_sha256": digest("sealed-release"),
        "seed": 9,
        "seed_collections": seed_rows,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "study_lock": lock_ref,
    }


def _receipt_ref_4d(value: dict[str, object], *, payload: bytes | None = None):
    body = payload if payload is not None else canonical_receipt_bytes(value)
    sha256 = hashlib.sha256(body).hexdigest()
    return body, {
        "uri": f"{S3_ROOT}/" + cohort_collection_receipt_key(sha256),
        "sha256": sha256,
        "version_id": "cohort-collection-version-1",
    }


def _terminal_binding(*, boot_id: str | None = None):
    values = lifecycle_core()
    if boot_id is not None:
        values = lifecycle_core(boot_id=boot_id)
    return expected_lifecycle_binding(
        default_provider_selection(),
        9,
        core=values,
    )


# ---------------------------------------------------------------------------
# Module contracts (brief tests 2, 15).
# ---------------------------------------------------------------------------


def test_cohort_collection_contracts_are_exact():
    assert COHORT_COLLECTION_RECEIPT_TYPE == (
        "memorysplit-aws-cohort-collection-v3"
    )
    assert COHORT_COLLECTION_OBJECT_COUNT == 1_012
    assert type(COHORT_COLLECTION_OBJECT_COUNT) is int
    assert issubclass(CohortCollectionError, ValueError)


def test_cohort_receipt_fields_mirror_frozen_3f_and_3d_sets():
    from cluster.aws.p5.run_finalization import _RECEIPT_FIELDS
    from msctl.aws_collect import COLLECTION_RECEIPT_FIELDS

    assert COHORT_COLLECTION_RECEIPT_FIELDS == frozenset(
        (COLLECTION_RECEIPT_FIELDS - {"run_receipt", "checkpoint_receipt"})
        | {
            "study_lock",
            "cohort_report",
            "seed_collections",
            "sealed_evaluation_release_sha256",
            "preregistration_sha256",
        }
    )
    assert COLLECTION_RECEIPT_FIELDS == frozenset(
        (_RECEIPT_FIELDS - {"arms", "finalized_at"})
        | {"collected_at", "objects", "run_receipt"}
    )
    assert len(COHORT_COLLECTION_RECEIPT_FIELDS) == 37


def test_cohort_shape_patterns_mirror_the_frozen_3f_module():
    # The cohort module cannot import the Task 3F module at module scope
    # (that chain pulls torch through the frozen Task 3D finalization
    # imports), so its shape patterns are duplicated and pinned here.
    import msctl.aws_cohort_collect as cohort_module
    import msctl.aws_collect as seed_module

    for name in (
        "_SHA256_RE",
        "_REQUEST_ID_RE",
        "_GIT_SHA1_RE",
        "_UTC_RE",
        "_INSTANCE_RE",
        "_BOOT_RE",
        "_BUCKET_RE",
    ):
        assert getattr(cohort_module, name).pattern == (
            getattr(seed_module, name).pattern
        ), name
    assert cohort_module._PROFILE_PROVIDERS == seed_module._PROFILE_PROVIDERS


def test_member_byte_ceiling_mirrors_aggregate_constant():
    import evals.confirmatory.aggregate as aggregate_module
    from msctl.aws_cohort_collect import _MAX_EVIDENCE_OBJECT_BYTES

    assert _MAX_EVIDENCE_OBJECT_BYTES == (
        aggregate_module._MAX_OUTPUT_FILE_BYTES
    )
    assert _MAX_EVIDENCE_OBJECT_BYTES == _MEMBER_CEILING


def test_frozen_slot_order_matches_study_lock_registry():
    from evals.confirmatory.study_lock import EXPECTED_STUDY_SLOTS_V3

    assert tuple(
        (seed, arm.value, step) for seed, arm, step in EXPECTED_STUDY_SLOTS_V3
    ) == _SLOTS


def test_cohort_dataclasses_fail_closed():
    reference = CohortEvidenceRef(
        uri="s3://bucket/key.json",
        sha256="a" * 64,
        version_id="v1",
        bytes=1,
    )
    assert reference.bytes == 1
    with pytest.raises(CohortCollectionError):
        CohortEvidenceRef(
            uri="https://bucket/key.json",
            sha256="a" * 64,
            version_id="v1",
            bytes=1,
        )
    with pytest.raises(CohortCollectionError):
        CohortEvidenceRef(
            uri="s3://bucket/key.json",
            sha256="A" * 64,
            version_id="v1",
            bytes=1,
        )
    with pytest.raises(CohortCollectionError):
        CohortEvidenceRef(
            uri="s3://bucket/key.json",
            sha256="a" * 64,
            version_id="null",
            bytes=1,
        )
    with pytest.raises(CohortCollectionError):
        CohortEvidenceRef(
            uri="s3://bucket/key.json",
            sha256="a" * 64,
            version_id="v1",
            bytes=0,
        )
    with pytest.raises(CohortCollectionError):
        CohortEvidenceRef(
            uri="s3://bucket/key.json",
            sha256="a" * 64,
            version_id="v1",
            bytes=True,
        )
    member = CohortCollectedObject(
        kind="output_member",
        seed=0,
        arm="dense",
        step=1_358,
        member="output.json",
        uri="s3://bucket/object.json",
        sha256="a" * 64,
        bytes=1,
        version_id="v1",
    )
    assert member.member == "output.json"
    with pytest.raises(CohortCollectionError):
        CohortCollectedObject(
            kind="artifact",
            seed=0,
            arm="dense",
            step=1_358,
            member="output.json",
            uri="s3://bucket/object.json",
            sha256="a" * 64,
            bytes=1,
            version_id="v1",
        )
    with pytest.raises(CohortCollectionError):
        CohortCollectedObject(
            kind="output_member",
            seed=0,
            arm="dense",
            step=1_358,
            member="weights.pt",
            uri="s3://bucket/object.json",
            sha256="a" * 64,
            bytes=1,
            version_id="v1",
        )
    with pytest.raises(CohortCollectionError):
        CohortCollectedObject(
            kind="output_member",
            seed=0,
            arm="dense",
            step=999,
            member="output.json",
            uri="s3://bucket/object.json",
            sha256="a" * 64,
            bytes=1,
            version_id="v1",
        )
    with pytest.raises(CohortCollectionError):
        CohortCollectedObject(
            kind="seed_collection",
            seed=3,
            arm="dense",
            step=None,
            member=None,
            uri="s3://bucket/object.json",
            sha256="a" * 64,
            bytes=1,
            version_id="v1",
        )
    with pytest.raises(CohortCollectionError):
        CohortCollectedObject(
            kind="study_lock",
            seed=1,
            arm=None,
            step=None,
            member=None,
            uri="s3://bucket/object.json",
            sha256="a" * 64,
            bytes=1,
            version_id="v1",
        )
    anchor = CohortCollectedObject(
        kind="cohort_report",
        seed=None,
        arm=None,
        step=None,
        member=None,
        uri="s3://bucket/object.json",
        sha256="a" * 64,
        bytes=1,
        version_id="v1",
    )
    assert anchor.kind == "cohort_report"


# ---------------------------------------------------------------------------
# Receipt parser matrix (brief test 3).
# ---------------------------------------------------------------------------


def test_parse_cohort_collection_receipt_roundtrip():
    binding = _terminal_binding()
    value = _synthetic_receipt_value(binding)
    payload, ref = _receipt_ref_4d(value)

    parsed = parse_cohort_collection_receipt_bytes(
        payload,
        receipt_uri=ref["uri"],
        receipt_sha256=ref["sha256"],
        receipt_version_id=ref["version_id"],
        expected_binding=binding,
    )

    assert parsed == value
    rows = parsed["objects"]
    assert len(rows) == COHORT_COLLECTION_OBJECT_COUNT
    assert [row["kind"] for row in rows] == [
        *(["output_member"] * 1_000),
        *(["seed_collection"] * 10),
        "study_lock",
        "cohort_report",
    ]
    assert [row["member"] for row in rows[:10]] == list(
        EVALUATION_OUTPUT_MEMBERS
    )
    assert rows[0]["seed"] == 0
    assert rows[0]["arm"] == "dense"
    assert rows[0]["step"] == SNAPSHOT_STEPS[0]
    assert rows[999]["seed"] == 9
    assert rows[999]["arm"] == "split90"
    assert rows[999]["step"] == SNAPSHOT_STEPS[-1]
    assert [row["seed"] for row in rows[1_000:1_010]] == list(SEEDS)
    for offset, (kind, reference) in enumerate(
        (
            ("study_lock", parsed["study_lock"]),
            ("cohort_report", parsed["cohort_report"]),
        )
    ):
        row = rows[1_010 + offset]
        assert row["kind"] == kind
        assert {
            field: row[field]
            for field in ("uri", "sha256", "bytes", "version_id")
        } == reference


def test_parser_accepts_boot_only_rebind_and_rejects_instance_drift():
    binding = _terminal_binding(boot_id=CONTROLLER_BOOT_ID)
    value = _synthetic_receipt_value(binding)
    payload, ref = _receipt_ref_4d(value)

    # A controller running on a later boot admits the receipt only after
    # rebinding the boot; every other commitment must match exactly.
    rebased = _terminal_binding()
    assert rebased.boot_id != CONTROLLER_BOOT_ID
    parse_cohort_collection_receipt_bytes(
        payload,
        receipt_uri=ref["uri"],
        receipt_sha256=ref["sha256"],
        receipt_version_id=ref["version_id"],
        expected_binding=replace(rebased, boot_id=CONTROLLER_BOOT_ID),
    )
    with pytest.raises(CohortCollectionError):
        parse_cohort_collection_receipt_bytes(
            payload,
            receipt_uri=ref["uri"],
            receipt_sha256=ref["sha256"],
            receipt_version_id=ref["version_id"],
            expected_binding=rebased,
        )
    with pytest.raises(CohortCollectionError):
        parse_cohort_collection_receipt_bytes(
            payload,
            receipt_uri=ref["uri"],
            receipt_sha256=ref["sha256"],
            receipt_version_id=ref["version_id"],
            expected_binding=replace(
                rebased,
                boot_id=CONTROLLER_BOOT_ID,
                instance_id=FOREIGN_INSTANCE_ID,
            ),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "appended-byte",
        "hash-mismatch",
        "null-receipt-version",
        "duplicate-key",
        "non-finite",
        "not-canonical-order",
        "empty-payload",
        "uri-wrong-digest",
        "uri-not-s3",
        "uri-unsafe-root",
        "extra-field",
        "missing-field",
        "schema-version",
        "receipt-type",
        "cohort-id",
        "complete-false",
        "seed-not-terminal",
        "request-id-shape",
        "collected-at-shape",
        "provider-profile-crossed",
        "source-commit-shape",
        "instance-shape",
        "boot-shape",
        "sha-field-invalid",
        "selection-version-null",
    ],
)
def test_cohort_receipt_identity_matrix_fails_closed(mutation):
    binding = _terminal_binding()
    value = _synthetic_receipt_value(binding)
    payload, ref = _receipt_ref_4d(value)
    receipt_uri = ref["uri"]
    receipt_sha256 = ref["sha256"]
    receipt_version_id = ref["version_id"]
    if mutation == "appended-byte":
        payload = payload + b"\n"
    elif mutation == "hash-mismatch":
        receipt_sha256 = "c" * 64
    elif mutation == "null-receipt-version":
        receipt_version_id = "null"
    elif mutation == "duplicate-key":
        payload = (
            b'{"seed":9,"seed":9' + payload[1 + payload.index(b","):]
        )
        receipt_sha256 = hashlib.sha256(payload).hexdigest()
        receipt_uri = f"{S3_ROOT}/" + cohort_collection_receipt_key(
            receipt_sha256
        )
    elif mutation == "non-finite":
        payload = payload.replace(b'"bytes":8192', b'"bytes":NaN', 1)
        receipt_sha256 = hashlib.sha256(payload).hexdigest()
        receipt_uri = f"{S3_ROOT}/" + cohort_collection_receipt_key(
            receipt_sha256
        )
    elif mutation == "not-canonical-order":
        decoded = json.loads(payload.decode("ascii"))
        reordered = dict(reversed(list(decoded.items())))
        payload = json.dumps(
            reordered,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii") + b"\n"
        receipt_sha256 = hashlib.sha256(payload).hexdigest()
        receipt_uri = f"{S3_ROOT}/" + cohort_collection_receipt_key(
            receipt_sha256
        )
    elif mutation == "empty-payload":
        payload = b""
    elif mutation == "uri-wrong-digest":
        receipt_uri = f"{S3_ROOT}/" + cohort_collection_receipt_key("c" * 64)
    elif mutation == "uri-not-s3":
        receipt_uri = receipt_uri.replace("s3://", "https://")
    elif mutation == "uri-unsafe-root":
        receipt_uri = (
            "s3://memorysplit-prod/../confirmatory-v3/"
            + cohort_collection_receipt_key(receipt_sha256)
        )
    else:
        if mutation == "extra-field":
            value["run_receipt"] = dict(value["seed_collections"][0])
        elif mutation == "missing-field":
            value.pop("collected_at")
        elif mutation == "schema-version":
            value["schema_version"] = 2
        elif mutation == "receipt-type":
            value["receipt_type"] = "memorysplit-aws-seed-collection-v3"
        elif mutation == "cohort-id":
            value["cohort_id"] = "memorysplit-confirmatory-v3-360m-n10-gcp"
        elif mutation == "complete-false":
            value["complete"] = False
        elif mutation == "seed-not-terminal":
            value["seed"] = 8
        elif mutation == "request-id-shape":
            value["request_id"] = "not-a-request"
        elif mutation == "collected-at-shape":
            value["collected_at"] = "2026-07-24 04:00:00"
        elif mutation == "provider-profile-crossed":
            value["profile_id"] = "aws-p6-b300.48xlarge-v3"
        elif mutation == "source-commit-shape":
            value["source_commit"] = "x" * 40
        elif mutation == "instance-shape":
            value["instance_id"] = "not-an-instance"
        elif mutation == "boot-shape":
            value["boot_id"] = "not-a-boot"
        elif mutation == "sha-field-invalid":
            value["release_sha256"] = "X" * 64
        elif mutation == "selection-version-null":
            value["provider_selection_version_id"] = "null"
        payload, ref = _receipt_ref_4d(value)
        receipt_uri = ref["uri"]
        receipt_sha256 = ref["sha256"]
        receipt_version_id = ref["version_id"]

    with pytest.raises(CohortCollectionError):
        parse_cohort_collection_receipt_bytes(
            payload,
            receipt_uri=receipt_uri,
            receipt_sha256=receipt_sha256,
            receipt_version_id=receipt_version_id,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-row",
        "extra-row",
        "reordered-members",
        "reordered-slots",
        "duplicate-row",
        "cross-slot-swap",
        "cross-receipt-swap",
        "foreign-kind",
        "member-on-seed-row",
        "seed-on-anchor-row",
        "row-fields-extra",
        "row-fields-missing",
        "row-bytes-zero",
        "row-bytes-bool",
        "row-version-null",
        "row-uri-key-mismatch",
        "seed-rows-reordered",
        "seed-row-aliased",
        "lock-row-differs-from-reference",
        "report-row-differs-from-reference",
        "seed-row-differs-from-reference",
        "top-lock-wrong-key",
        "top-report-wrong-key",
        "top-seed-wrong-seed-key",
        "objects-not-a-list",
    ],
)
def test_cohort_receipt_object_rows_are_exact(mutation):
    binding = _terminal_binding()
    value = _synthetic_receipt_value(binding)
    rows = value["objects"]
    if mutation == "missing-row":
        rows.pop(17)
    elif mutation == "extra-row":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "reordered-members":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "reordered-slots":
        rows[0:10], rows[10:20] = rows[10:20], rows[0:10]
    elif mutation == "duplicate-row":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "cross-slot-swap":
        rows[4] = copy.deepcopy(rows[14])
    elif mutation == "cross-receipt-swap":
        # A row lifted from a different receipt keeps its slot shape but
        # carries a foreign content hash, so its canonical key cannot
        # match this receipt's URI.
        rows[3] = {
            **copy.deepcopy(rows[3]),
            "sha256": digest("foreign-member"),
        }
    elif mutation == "foreign-kind":
        rows[0]["kind"] = "snapshot"
    elif mutation == "member-on-seed-row":
        rows[1_000]["member"] = "output.json"
    elif mutation == "seed-on-anchor-row":
        rows[1_010]["seed"] = 9
    elif mutation == "row-fields-extra":
        rows[0]["etag"] = "abc"
    elif mutation == "row-fields-missing":
        rows[0].pop("version_id")
    elif mutation == "row-bytes-zero":
        rows[0]["bytes"] = 0
    elif mutation == "row-bytes-bool":
        rows[0]["bytes"] = True
    elif mutation == "row-version-null":
        rows[0]["version_id"] = "null"
    elif mutation == "row-uri-key-mismatch":
        rows[0]["uri"] = str(rows[0]["uri"]).replace(
            "step-1358",
            "step-3396",
        )
    elif mutation == "seed-rows-reordered":
        rows[1_000], rows[1_001] = rows[1_001], rows[1_000]
        value["seed_collections"][0], value["seed_collections"][1] = (
            value["seed_collections"][1],
            value["seed_collections"][0],
        )
    elif mutation == "seed-row-aliased":
        aliased = digest("synthetic-seed-collection:0")
        value["seed_collections"][1]["sha256"] = aliased
        value["seed_collections"][1]["uri"] = (
            f"{S3_ROOT}/" + collection_receipt_key(1, aliased)
        )
        rows[1_001]["sha256"] = aliased
        rows[1_001]["uri"] = value["seed_collections"][1]["uri"]
    elif mutation == "lock-row-differs-from-reference":
        rows[1_010]["bytes"] = rows[1_010]["bytes"] + 1
    elif mutation == "report-row-differs-from-reference":
        rows[1_011]["version_id"] = "cohort-report-version-2"
    elif mutation == "seed-row-differs-from-reference":
        rows[1_003]["version_id"] = "collection-version-drifted"
    elif mutation == "top-lock-wrong-key":
        value["study_lock"]["uri"] = (
            f"{S3_ROOT}/"
            + cohort_report_object_key(value["study_lock"]["sha256"])
        )
        rows[1_010]["uri"] = value["study_lock"]["uri"]
    elif mutation == "top-report-wrong-key":
        value["cohort_report"]["uri"] = (
            f"{S3_ROOT}/"
            + study_lock_object_key(value["cohort_report"]["sha256"])
        )
        rows[1_011]["uri"] = value["cohort_report"]["uri"]
    elif mutation == "top-seed-wrong-seed-key":
        moved = value["seed_collections"][2]
        moved["uri"] = f"{S3_ROOT}/" + collection_receipt_key(
            3,
            moved["sha256"],
        )
        rows[1_002]["uri"] = moved["uri"]
    elif mutation == "objects-not-a-list":
        value["objects"] = {"rows": rows}
    payload, ref = _receipt_ref_4d(value)

    with pytest.raises(CohortCollectionError):
        parse_cohort_collection_receipt_bytes(
            payload,
            receipt_uri=ref["uri"],
            receipt_sha256=ref["sha256"],
            receipt_version_id=ref["version_id"],
        )


def test_cohort_receipt_carries_no_outcome_fields():
    assert not (
        COHORT_COLLECTION_RECEIPT_FIELDS
        & {
            "primary",
            "aulc",
            "secondary",
            "status",
            "final_inference_conclusion",
            "statistic",
            "conclusion",
        }
    )


# ---------------------------------------------------------------------------
# Evidence-index matrix (brief test 4).
# ---------------------------------------------------------------------------


def test_parse_cohort_evidence_index_roundtrip():
    value = _synthetic_index()
    parsed = parse_cohort_evidence_index(
        canonical_receipt_bytes(value),
        s3_root=S3_ROOT,
    )
    assert parsed == value
    assert [row["slot_index"] for row in parsed["outputs"]] == list(
        range(100)
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "not-json",
        "not-an-object",
        "duplicate-key",
        "non-finite",
        "schema-version",
        "cohort-id",
        "extra-field",
        "missing-field",
        "outputs-99",
        "outputs-101",
        "outputs-duplicate-slot",
        "outputs-reordered",
        "members-9",
        "members-11",
        "members-reordered",
        "member-key-mismatch",
        "member-foreign-root",
        "member-bytes-over-ceiling",
        "member-bytes-zero",
        "member-version-null",
        "output-id-drift",
        "output-id-foreign-profile",
        "slot-index-drift",
        "seed-rows-9",
        "seed-rows-11",
        "seed-rows-reordered",
        "seed-row-key-mismatch",
        "seed-row-aliased",
        "lock-key-mismatch",
        "report-key-mismatch",
        "lock-bytes-over-ceiling",
        "unsafe-root-argument",
    ],
)
def test_evidence_index_matrix_fails_closed(mutation):
    value = _synthetic_index()
    s3_root = S3_ROOT
    payload: bytes | None = None
    if mutation == "not-json":
        payload = b"not json"
    elif mutation == "not-an-object":
        payload = b"[1,2,3]\n"
    elif mutation == "duplicate-key":
        payload = (
            b'{"schema_version":1,"schema_version":1'
            + canonical_receipt_bytes(value)[
                1 + canonical_receipt_bytes(value).index(b","):
            ]
        )
    elif mutation == "non-finite":
        payload = canonical_receipt_bytes(value).replace(
            b'"bytes":8192',
            b'"bytes":Infinity',
            1,
        )
    elif mutation == "schema-version":
        value["schema_version"] = 2
    elif mutation == "cohort-id":
        value["cohort_id"] = "memorysplit-confirmatory-v3-360m-n10-gcp"
    elif mutation == "extra-field":
        value["publisher"] = "task-4e"
    elif mutation == "missing-field":
        value.pop("study_lock")
    elif mutation == "outputs-99":
        value["outputs"].pop()
    elif mutation == "outputs-101":
        value["outputs"].append(copy.deepcopy(value["outputs"][-1]))
    elif mutation == "outputs-duplicate-slot":
        value["outputs"][1] = copy.deepcopy(value["outputs"][0])
    elif mutation == "outputs-reordered":
        value["outputs"][0], value["outputs"][1] = (
            value["outputs"][1],
            value["outputs"][0],
        )
    elif mutation == "members-9":
        value["outputs"][0]["members"].pop()
    elif mutation == "members-11":
        value["outputs"][0]["members"].append(
            copy.deepcopy(value["outputs"][0]["members"][-1])
        )
    elif mutation == "members-reordered":
        members = value["outputs"][0]["members"]
        members[0], members[1] = members[1], members[0]
    elif mutation == "member-key-mismatch":
        row = value["outputs"][0]["members"][0]
        row["uri"] = f"{S3_ROOT}/" + evaluation_output_member_key(
            0,
            "dense",
            1_358,
            "inference.json",
            "c" * 64,
        )
    elif mutation == "member-foreign-root":
        row = value["outputs"][0]["members"][0]
        row["uri"] = str(row["uri"]).replace(
            "memorysplit-prod",
            "foreign-bucket",
        )
    elif mutation == "member-bytes-over-ceiling":
        value["outputs"][0]["members"][0]["bytes"] = _MEMBER_CEILING + 1
    elif mutation == "member-bytes-zero":
        value["outputs"][0]["members"][0]["bytes"] = 0
    elif mutation == "member-version-null":
        value["outputs"][0]["members"][0]["version_id"] = "null"
    elif mutation == "output-id-drift":
        value["outputs"][0]["output_id"] = _output_id(0, "dense", 3_396)
    elif mutation == "output-id-foreign-profile":
        for row in value["outputs"]:
            row["output_id"] = row["output_id"].replace(
                PROFILE_ID,
                "aws-h200.48xlarge-v9",
            )
    elif mutation == "slot-index-drift":
        value["outputs"][5]["slot_index"] = 6
    elif mutation == "seed-rows-9":
        value["seed_collections"].pop()
    elif mutation == "seed-rows-11":
        value["seed_collections"].append(
            copy.deepcopy(value["seed_collections"][-1])
        )
    elif mutation == "seed-rows-reordered":
        value["seed_collections"][0], value["seed_collections"][1] = (
            value["seed_collections"][1],
            value["seed_collections"][0],
        )
    elif mutation == "seed-row-key-mismatch":
        row = value["seed_collections"][0]
        row["uri"] = f"{S3_ROOT}/" + collection_receipt_key(0, "c" * 64)
    elif mutation == "seed-row-aliased":
        aliased = value["seed_collections"][0]["sha256"]
        value["seed_collections"][1]["sha256"] = aliased
        value["seed_collections"][1]["uri"] = (
            f"{S3_ROOT}/" + collection_receipt_key(1, aliased)
        )
    elif mutation == "lock-key-mismatch":
        value["study_lock"]["uri"] = (
            f"{S3_ROOT}/" + study_lock_object_key("c" * 64)
        )
    elif mutation == "report-key-mismatch":
        value["cohort_report"]["uri"] = (
            f"{S3_ROOT}/" + cohort_report_object_key("c" * 64)
        )
    elif mutation == "lock-bytes-over-ceiling":
        value["study_lock"]["bytes"] = _MEMBER_CEILING + 1
    elif mutation == "unsafe-root-argument":
        s3_root = "s3://memorysplit-prod/../confirmatory-v3"

    body = payload if payload is not None else canonical_receipt_bytes(value)
    with pytest.raises(CohortCollectionError):
        parse_cohort_evidence_index(body, s3_root=s3_root)


# ---------------------------------------------------------------------------
# Payload-real cohort wrapped as an S3 evidence fixture.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    from tests.cohort_output_fixtures import _build_cohort

    return _build_cohort(tmp_path_factory.mktemp("cohort-4d"), "effect")


@pytest.fixture(scope="module")
def s3_cohort(cohort):
    return _wrap_cohort(cohort)


def _wrap_cohort(cohort):
    from evals.confirmatory.aggregate import aggregate_cohort_outputs
    from evals.confirmatory.contracts import canonical_json_bytes

    report = aggregate_cohort_outputs(
        outputs=cohort.references,
        collection_receipts=cohort.receipts,
        expected_study_lock_sha256=cohort.lock_sha256,
    )
    report_bytes = report.canonical_bytes
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    lock_bytes = canonical_json_bytes(cohort.lock.to_dict())
    bodies: dict[str, tuple[bytes, str]] = {}
    lock_ref = {
        "uri": f"{S3_ROOT}/" + study_lock_object_key(cohort.lock_sha256),
        "sha256": cohort.lock_sha256,
        "version_id": "study-lock-version-1",
        "bytes": len(lock_bytes),
    }
    bodies[lock_ref["uri"]] = (lock_bytes, lock_ref["version_id"])
    report_ref = {
        "uri": f"{S3_ROOT}/" + cohort_report_object_key(report_sha256),
        "sha256": report_sha256,
        "version_id": "cohort-report-version-1",
        "bytes": len(report_bytes),
    }
    bodies[report_ref["uri"]] = (report_bytes, report_ref["version_id"])
    seed_rows = []
    for seed, evidence in zip(SEEDS, cohort.receipts, strict=True):
        seed_rows.append(
            {
                "seed": seed,
                "uri": evidence.uri,
                "sha256": evidence.sha256,
                "version_id": evidence.version_id,
                "bytes": len(evidence.payload),
            }
        )
        bodies[evidence.uri] = (evidence.payload, evidence.version_id)
    outputs = []
    for index, ((seed, arm, step), reference) in enumerate(
        zip(_SLOTS, cohort.references, strict=True)
    ):
        members = []
        for member in EVALUATION_OUTPUT_MEMBERS:
            content = (reference.output_dir / member).read_bytes()
            sha256 = hashlib.sha256(content).hexdigest()
            uri = f"{S3_ROOT}/" + evaluation_output_member_key(
                seed,
                arm,
                step,
                member,
                sha256,
            )
            version_id = f"member-version-{index:03d}-{member}"
            members.append(
                {
                    "member": member,
                    "uri": uri,
                    "sha256": sha256,
                    "version_id": version_id,
                    "bytes": len(content),
                }
            )
            bodies[uri] = (content, version_id)
        outputs.append(
            {
                "slot_index": index,
                "seed": seed,
                "arm": arm,
                "optimizer_step": step,
                "output_id": report.inputs[index]["output_id"],
                "members": members,
            }
        )
    index_value = {
        "schema_version": 1,
        "cohort_id": COHORT_ID,
        "study_lock": lock_ref,
        "cohort_report": report_ref,
        "seed_collections": seed_rows,
        "outputs": outputs,
    }
    total_bytes = (
        sum(
            row["bytes"]
            for output in outputs
            for row in output["members"]
        )
        + sum(row["bytes"] for row in seed_rows)
        + lock_ref["bytes"]
        + report_ref["bytes"]
    )
    return SimpleNamespace(
        cohort=cohort,
        report=report,
        report_bytes=report_bytes,
        report_ref=report_ref,
        lock_bytes=lock_bytes,
        lock_ref=lock_ref,
        seed_rows=seed_rows,
        index_value=index_value,
        bodies=bodies,
        total_bytes=total_bytes,
        sealed_release_sha256=(
            cohort.lock.sealed_evaluation_release_sha256
        ),
        preregistration_sha256=cohort.lock.preregistration_sha256,
    )


def _cohort_lifecycle(
    profile,
    *,
    seed: int,
    boot_id: str | None = None,
    instance_id: str | None = None,
):
    from cluster.aws.qualification import CohortSelectionAuthority
    from msctl.aws_hardware import AuthenticatedSelectionBinding
    from msctl.aws_lifecycle import AuthenticatedProviderLifecycle

    selection = default_provider_selection()
    values = lifecycle_core()
    if boot_id is not None:
        values["boot_id"] = boot_id
    if instance_id is not None:
        values["instance_id"] = instance_id
    binding = expected_lifecycle_binding(selection, seed, core=values)
    dense = AuthenticatedSelectionBinding(
        cohort_id=selection["cohort_id"],
        amendment_sha256=selection["hardware_amendment_sha256"],
        selection_sha256=selection["provider_selection_sha256"],
        selection_version_id=selection["provider_selection_s3_version_id"],
        profile_id=profile.profile_id,
        provider=profile.provider,
        profile_sha256=profile.sha256,
        runtime_lock_sha256=selection["runtime_lock_sha256"],
        qualification_evidence_sha256=selection[
            "qualification_evidence_sha256"
        ],
        environment_receipt_sha256=selection["environment_receipt_sha256"],
        canary_receipt_sha256=selection["canary_receipt_sha256"],
        approval_receipt_sha256=selection["approval_receipt_sha256"],
        approval_public_key_sha256=selection["approval_public_key_sha256"],
        account_id=values["account_id"],
        instance_id=values["instance_id"],
        boot_id=values["boot_id"],
        region=values["region"],
        availability_zone=values["availability_zone"],
        purchase_model=values["purchase_model"],
        seed=seed,
        arm="dense",
    )
    authority = CohortSelectionAuthority(
        profile=profile,
        seed=seed,
        arms=("dense", "split90"),
        bindings={"dense": dense, "split90": replace(dense, arm="split90")},
    )
    return AuthenticatedProviderLifecycle(
        binding=binding,
        profile=authority.profile,
        arm_bindings=authority.bindings,
    )


def _cohort_backend(
    tmp_path,
    set_module_attr,
    *,
    seed: int = 9,
    boot_id: str | None = CONTROLLER_BOOT_ID,
    instance_id: str | None = None,
):
    import msctl.aws_p5 as module

    profile = load_aws_gpu_profile(P5_PROFILE)
    set_module_attr(
        module,
        "admit_provider_lifecycle",
        lambda **kwargs: _cohort_lifecycle(
            profile,
            seed=kwargs["seed"],
            boot_id=boot_id,
            instance_id=instance_id,
        ),
    )
    lifecycle = _cohort_lifecycle(
        profile,
        seed=seed,
        boot_id=boot_id,
        instance_id=instance_id,
    )
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


def _cohort_manifest(
    binding,
    *,
    sealed_release_sha256: str,
    preregistration_sha256: str,
):
    seed = binding.seed
    runs = tuple(
        SimpleNamespace(
            run_id=f"memorysplit-v3-360m-s{seed}-{arm}",
            arm=arm,
            seed=seed,
            config=f"configs/360m-v3/{arm}-s{seed}.yaml",
            config_sha256=("a" if arm == "dense" else "b") * 64,
        )
        for arm in ARMS
    )
    return SimpleNamespace(
        **binding.to_dict(),
        schema_version=3,
        release_sha256=RELEASE_SHA256,
        release_receipt_sha256=RELEASE_RECEIPT_SHA256,
        dataset_pointer_sha256=digest("dataset-pointer"),
        dataset_receipt_sha256=digest("data-receipt"),
        dataset_build_id=digest("data-build"),
        ordered_stream_sha256=digest("ordered-stream"),
        cohort_assignment_sha256=digest("cohort-assignment"),
        preregistration_sha256=preregistration_sha256,
        sealed_evaluation_release_sha256=sealed_release_sha256,
        source_commit=SOURCE_COMMIT,
        source_tree=SOURCE_TREE,
        sha256=digest(f"run-manifest-{seed}"),
        runs=runs,
    )


def _cohort_release(manifest):
    return SimpleNamespace(
        provider=manifest.provider,
        archive_sha256=manifest.release_sha256,
        receipt_sha256=manifest.release_receipt_sha256,
        members_sha256="6" * 64,
        source_commit=manifest.source_commit,
        source_tree=manifest.source_tree,
    )


def _collect_setup(
    tmp_path,
    set_module_attr,
    s3,
    *,
    seed: int = 9,
    boot_id: str | None = CONTROLLER_BOOT_ID,
    instance_id: str | None = None,
    index_value: dict | None = None,
    bodies: dict[str, tuple[bytes, str]] | None = None,
    anchor: dict[str, str] | None = None,
):
    backend, lifecycle, runner = _cohort_backend(
        tmp_path,
        set_module_attr,
        seed=seed,
        boot_id=boot_id,
        instance_id=instance_id,
    )
    downloads = _ScriptedDownloadRunner(
        dict(s3.bodies if bodies is None else bodies),
        runner,
    )
    backend.download_runner = downloads
    manifest = _cohort_manifest(
        lifecycle.binding,
        sealed_release_sha256=s3.sealed_release_sha256,
        preregistration_sha256=s3.preregistration_sha256,
    )
    release = _cohort_release(manifest)
    chosen_index = s3.index_value if index_value is None else index_value
    index_path = tmp_path / "cohort-evidence-index.json"
    index_path.write_bytes(canonical_receipt_bytes(chosen_index))
    report_row = chosen_index["cohort_report"]
    chosen_anchor = anchor or {
        "uri": report_row["uri"],
        "sha256": report_row["sha256"],
        "version_id": report_row["version_id"],
    }
    return SimpleNamespace(
        backend=backend,
        binding=lifecycle.binding,
        runner=runner,
        downloads=downloads,
        manifest=manifest,
        release=release,
        index_path=index_path,
        index_value=chosen_index,
        anchor=chosen_anchor,
        s3=s3,
    )


def _publish_cohort_outputs(record: dict[str, object]):
    def _put(argv: list[str]):
        argv = list(argv)
        record["put_argv"] = argv
        checksum = argv[argv.index("--checksum-sha256") + 1]
        record["checksum"] = checksum
        record["metadata"] = dict(
            part.split("=", 1)
            for part in argv[argv.index("--metadata") + 1].split(",")
        )
        record["body"] = Path(argv[argv.index("--body") + 1]).read_bytes()
        return {
            "object": {
                "checksum_sha256": checksum,
                "version_id": "cohort-collection-version-1",
            }
        }

    def _head(argv: list[str]):
        del argv
        return {
            "object": {
                "checksum_sha256": record["checksum"],
                "content_length": len(record["body"]),
                "metadata": record["metadata"],
                "version_id": "cohort-collection-version-1",
            }
        }

    return _put, _head


def _collect_cohort(setup, out, *, apply: bool = True):
    return setup.backend.collect_cohort_evidence(
        release=setup.release,
        manifest=setup.manifest,
        cohort_report=setup.anchor,
        evidence_index=setup.index_path,
        out=out,
        apply=apply,
    )


def _assert_no_mutation(setup, out, *, state_root=None):
    assert not out.exists()
    assert not any(
        path.name.startswith(".") and "collect-cohort" in path.name
        for path in out.parent.iterdir()
    )
    assert not any(
        "put-object" in argv for argv, _operation in setup.runner.calls
    )
    from msctl.state import StateStore

    store = StateStore(state_root or setup.backend.state_root)
    with store.locked():
        assert store.read_cohort_collection() is None


# ---------------------------------------------------------------------------
# Full flow (brief exact enumeration; boot-only rebind is inherent: the
# controller boot differs from the training boot recorded in the lock).
# ---------------------------------------------------------------------------


def test_collect_cohort_full_flow_is_ordered_durable_and_boot_rebound(
    tmp_path,
    monkeypatch,
    s3_cohort,
):
    from msctl.aws_collect import download_timeout_seconds
    from msctl.state import StateStore

    setup = _collect_setup(tmp_path, monkeypatch.setattr, s3_cohort)
    assert setup.binding.boot_id == CONTROLLER_BOOT_ID
    assert setup.binding.boot_id != BOOT_ID
    out = tmp_path / "cohort-evidence"
    record: dict[str, object] = {}
    put, head = _publish_cohort_outputs(record)
    setup.runner.outputs = [put, head]

    result = _collect_cohort(setup, out)

    assert result["provider"] == "aws-p5.48xlarge"
    assert result["seed"] == 9
    assert result["run_manifest_sha256"] == setup.manifest.sha256
    assert result["objects_collected"] == 1_012
    assert result["collected"] == 1_012
    assert result["idempotent"] is False
    assert result["out"] == str(out)
    assert result["bytes_collected"] == s3_cohort.total_bytes
    assert result["study_lock"] == s3_cohort.lock_ref
    assert result["cohort_report"] == s3_cohort.report_ref

    # Exact download enumeration: report, lock, ten receipts, then per
    # slot output.json followed by its nine artifacts.
    assert len(setup.downloads.calls) == 1_012
    fetched: list[str] = []
    for call in setup.downloads.calls:
        argv = call["argv"]
        bucket = argv[argv.index("--bucket") + 1]
        key = argv[argv.index("--key") + 1]
        uri = f"s3://{bucket}/{key}"
        fetched.append(uri)
        body, _version = s3_cohort.bodies[uri]
        assert call["timeout_seconds"] == download_timeout_seconds(len(body))
        assert "--version-id" in argv
        assert "--checksum-mode" in argv
    expected_order = [
        s3_cohort.report_ref["uri"],
        s3_cohort.lock_ref["uri"],
        *[row["uri"] for row in s3_cohort.seed_rows],
    ]
    for output in s3_cohort.index_value["outputs"]:
        by_member = {row["member"]: row for row in output["members"]}
        expected_order.append(by_member["output.json"]["uri"])
        expected_order.extend(
            by_member[member]["uri"]
            for member in EVALUATION_OUTPUT_MEMBERS
            if member != "output.json"
        )
    assert fetched == expected_order

    operations = [operation for _argv, operation in setup.runner.calls]
    assert operations == [
        "publish cohort collection receipt",
        "verify cohort collection receipt",
    ]
    assert "--if-none-match" in record["put_argv"]
    receipt_bytes = record["body"]
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    assert record["checksum"] == _checksum(receipt_sha256)
    assert record["metadata"] == {
        "receipt-sha256": receipt_sha256,
        "receipt-type": COHORT_COLLECTION_RECEIPT_TYPE,
        "request-id": record["metadata"]["request-id"],
    }
    assert result["cohort_collection_receipt"] == {
        "uri": f"{S3_ROOT}/" + cohort_collection_receipt_key(receipt_sha256),
        "sha256": receipt_sha256,
        "version_id": "cohort-collection-version-1",
        "bytes": len(receipt_bytes),
    }

    parsed = parse_cohort_collection_receipt_bytes(
        receipt_bytes,
        receipt_uri=result["cohort_collection_receipt"]["uri"],
        receipt_sha256=receipt_sha256,
        receipt_version_id="cohort-collection-version-1",
        expected_binding=setup.binding,
    )
    assert parsed["seed"] == 9
    assert parsed["run_manifest_sha256"] == setup.manifest.sha256
    assert parsed["sealed_evaluation_release_sha256"] == (
        s3_cohort.sealed_release_sha256
    )
    assert parsed["preregistration_sha256"] == (
        s3_cohort.preregistration_sha256
    )
    assert parsed["study_lock"] == s3_cohort.lock_ref
    assert parsed["cohort_report"] == s3_cohort.report_ref
    assert parsed["seed_collections"] == s3_cohort.seed_rows
    rows = parsed["objects"]
    assert len(rows) == 1_012
    assert rows[1_010]["sha256"] == s3_cohort.lock_ref["sha256"]
    assert rows[1_011]["sha256"] == s3_cohort.report_ref["sha256"]
    assert sum(row["bytes"] for row in rows) == result["bytes_collected"]

    # Destination layout: replay-shaped outputs plus published-shaped
    # report/lock directories and the receipt copy.
    report_dir = out / f"cohort-report-{s3_cohort.report_ref['sha256']}"
    assert (report_dir / "cohort-report.json").read_bytes() == (
        s3_cohort.report_bytes
    )
    assert stat.S_IMODE(
        (report_dir / "cohort-report.json").stat().st_mode
    ) == 0o444
    lock_dir = out / f"study-lock-{s3_cohort.lock_ref['sha256']}"
    assert (lock_dir / "study-lock.json").read_bytes() == s3_cohort.lock_bytes
    assert stat.S_IMODE(
        (lock_dir / "study-lock.json").stat().st_mode
    ) == 0o444
    for output in s3_cohort.index_value["outputs"]:
        output_dir = out / "outputs" / output["output_id"]
        assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
        names = sorted(path.name for path in output_dir.iterdir())
        assert tuple(names) == EVALUATION_OUTPUT_MEMBERS
        for row in output["members"]:
            member_path = output_dir / row["member"]
            assert stat.S_IMODE(member_path.stat().st_mode) == 0o600
            assert member_path.stat().st_size == row["bytes"]
    for row in s3_cohort.seed_rows:
        local = (
            out
            / "seed-collections"
            / f"seed-{row['seed']}"
            / "sha256"
            / f"{row['sha256']}.json"
        )
        assert hashlib.sha256(local.read_bytes()).hexdigest() == row["sha256"]
    receipt_local = (
        out
        / "receipts"
        / "cohort-collections"
        / "sha256"
        / f"{receipt_sha256}.json"
    )
    assert receipt_local.read_bytes() == receipt_bytes
    assert not any(
        path.name.startswith(".") and "collect-cohort" in path.name
        for path in out.parent.iterdir()
    )

    store = StateStore(setup.backend.state_root)
    with store.locked():
        state = store.read_cohort_collection()
    assert state is not None
    assert state["operation"] == "collect-cohort"
    assert state["seed"] == 9
    assert state["status"] == "Published"
    assert state["objects_collected"] == 1_012
    assert state["bytes_collected"] == result["bytes_collected"]
    assert state["study_lock"] == s3_cohort.lock_ref
    assert state["cohort_report"] == s3_cohort.report_ref
    assert state["cohort_collection_receipt"] == (
        result["cohort_collection_receipt"]
    )
    assert state["sealed_evaluation_release_sha256"] == (
        s3_cohort.sealed_release_sha256
    )
    assert state["preregistration_sha256"] == (
        s3_cohort.preregistration_sha256
    )
    assert state["boot_id"] == CONTROLLER_BOOT_ID


def test_collect_cohort_dry_run_renders_first_report_get_only(
    tmp_path,
    monkeypatch,
    s3_cohort,
):
    setup = _collect_setup(tmp_path, monkeypatch.setattr, s3_cohort)
    out = tmp_path / "cohort-evidence"

    result = _collect_cohort(setup, out, apply=False)

    assert result["provider"] == "aws-p5.48xlarge"
    assert result["seed"] == 9
    assert result["collected"] == 0
    assert result["idempotent"] is False
    assert result["out"] == str(out)
    assert result["cohort_report"] == setup.anchor
    assert len(result["commands"]) == 1
    first = result["commands"][0]
    joined = " ".join(first)
    assert "get-object" in first
    assert "--version-id" in first
    assert setup.anchor["version_id"] in first
    assert "--checksum-mode" in first
    assert setup.anchor["sha256"] in joined
    assert setup.runner.calls == []
    assert setup.downloads.calls == []
    assert not out.exists()


# ---------------------------------------------------------------------------
# Gate failures precede every mutation (brief tests 5 and 11).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutation", "expected_code", "expected_downloads"),
    [
        ("missing-triple", "COHORT_COLLECT_EVIDENCE_INVALID", 0),
        ("malformed-sha", "COHORT_COLLECT_EVIDENCE_INVALID", 0),
        ("foreign-root-anchor", "COHORT_COLLECT_EVIDENCE_INVALID", 0),
        ("wrong-key-anchor", "COHORT_COLLECT_EVIDENCE_INVALID", 0),
        ("index-missing", "COHORT_COLLECT_EVIDENCE_INVALID", 0),
        ("index-invalid", "COHORT_COLLECT_EVIDENCE_INVALID", 0),
        ("index-anchor-drift", "COHORT_COLLECT_EVIDENCE_INVALID", 0),
        ("destination-exists", "COHORT_COLLECT_DESTINATION_EXISTS", 0),
        ("state-conflict", "COHORT_COLLECT_CONFLICT", 0),
        ("report-body-drift", "COHORT_COLLECT_OBJECT_MISMATCH", 1),
        ("report-checksum-drift", "COHORT_COLLECT_OBJECT_MISMATCH", 1),
        ("report-lock-sha-vs-index", "COHORT_COLLECT_EVIDENCE_INVALID", 1),
        (
            "report-receipt-row-vs-index",
            "COHORT_COLLECT_EVIDENCE_INVALID",
            1,
        ),
        ("sealed-release-vs-manifest", "COHORT_COLLECT_EVIDENCE_INVALID", 1),
        ("lock-body-drift", "COHORT_COLLECT_OBJECT_MISMATCH", 2),
        ("seed-receipt-body-drift", "COHORT_COLLECT_OBJECT_MISMATCH", 3),
        ("seed-receipt-identity-drift", "COHORT_COLLECT_EVIDENCE_INVALID", 2),
        ("output-commitment-drift", "COHORT_COLLECT_OBJECT_MISMATCH", 13),
        ("artifact-row-drift", "COHORT_COLLECT_OBJECT_MISMATCH", 13),
        ("output-unavailable", "COHORT_COLLECT_INCOMPLETE", 13),
    ],
)
def test_collect_cohort_gate_failures_precede_every_mutation(
    tmp_path,
    monkeypatch,
    s3_cohort,
    mutation,
    expected_code,
    expected_downloads,
):
    index_value = copy.deepcopy(s3_cohort.index_value)
    bodies = dict(s3_cohort.bodies)
    anchor = None
    setup = None
    if mutation == "index-anchor-drift":
        anchor = {
            "uri": s3_cohort.report_ref["uri"],
            "sha256": s3_cohort.report_ref["sha256"],
            "version_id": "cohort-report-version-2",
        }
    elif mutation == "index-invalid":
        index_value["outputs"].pop()
    elif mutation == "report-lock-sha-vs-index":
        drifted = digest("drifted-lock")
        index_value["study_lock"]["sha256"] = drifted
        index_value["study_lock"]["uri"] = (
            f"{S3_ROOT}/" + study_lock_object_key(drifted)
        )
    elif mutation == "report-receipt-row-vs-index":
        index_value["seed_collections"][4]["version_id"] = (
            "collection-version-drifted"
        )
    elif mutation == "seed-receipt-identity-drift":
        # A dishonestly republished report and index agree on a drifted
        # receipt identity, so the drift is only caught against the lock's
        # seed lifecycles after the second download.
        dishonest = json.loads(s3_cohort.report_bytes)
        dishonest["collection_receipts"][4]["version_id"] = (
            "collection-version-drifted"
        )
        _report_ref, index_value, bodies = _republish_report(
            s3_cohort,
            dishonest,
        )
        index_value["seed_collections"][4]["version_id"] = (
            "collection-version-drifted"
        )
    elif mutation == "output-commitment-drift":
        content = b'{"forged": "output manifest"}\n'
        sha256 = hashlib.sha256(content).hexdigest()
        uri = f"{S3_ROOT}/" + evaluation_output_member_key(
            0,
            "dense",
            SNAPSHOT_STEPS[0],
            "output.json",
            sha256,
        )
        row = index_value["outputs"][0]["members"][4]
        assert row["member"] == "output.json"
        row["sha256"] = sha256
        row["uri"] = uri
        row["bytes"] = len(content)
        bodies[uri] = (content, row["version_id"])
    elif mutation == "artifact-row-drift":
        content = b'{"forged": "metrics"}\n'
        sha256 = hashlib.sha256(content).hexdigest()
        uri = f"{S3_ROOT}/" + evaluation_output_member_key(
            0,
            "dense",
            SNAPSHOT_STEPS[0],
            "metrics.json",
            sha256,
        )
        row = index_value["outputs"][0]["members"][2]
        assert row["member"] == "metrics.json"
        row["sha256"] = sha256
        row["uri"] = uri
        row["bytes"] = len(content)
        bodies[uri] = (content, row["version_id"])

    setup = _collect_setup(
        tmp_path,
        monkeypatch.setattr,
        s3_cohort,
        index_value=index_value,
        bodies=bodies,
        anchor=anchor,
    )
    out = tmp_path / "cohort-evidence"
    triple = dict(setup.anchor)
    if mutation == "missing-triple":
        triple = None
    elif mutation == "malformed-sha":
        triple["sha256"] = "not-a-hash"
    elif mutation == "foreign-root-anchor":
        triple["uri"] = triple["uri"].replace(
            "memorysplit-prod",
            "foreign-bucket",
        )
    elif mutation == "wrong-key-anchor":
        triple["uri"] = f"{S3_ROOT}/" + cohort_report_object_key("c" * 64)
    elif mutation == "index-missing":
        setup.index_path.unlink()
    elif mutation == "destination-exists":
        out.mkdir(parents=True)
    elif mutation == "state-conflict":
        from msctl.state import StateStore

        conflicting = _cohort_state(
            setup.binding,
            study_lock=s3_cohort.lock_ref,
            cohort_report={
                **s3_cohort.report_ref,
                "sha256": digest("conflicting-report"),
                "uri": (
                    f"{S3_ROOT}/"
                    + cohort_report_object_key(digest("conflicting-report"))
                ),
            },
            run_manifest_sha256=setup.manifest.sha256,
            sealed_evaluation_release_sha256=(
                s3_cohort.sealed_release_sha256
            ),
            preregistration_sha256=s3_cohort.preregistration_sha256,
        )
        store = StateStore(setup.backend.state_root)
        with store.locked():
            store.write_cohort_collection(conflicting)
    elif mutation == "report-body-drift":
        setup.downloads.mutate_body[s3_cohort.report_ref["uri"]] = (
            s3_cohort.report_bytes[:-2] + b" \n"
        )
    elif mutation == "report-checksum-drift":
        setup.downloads.mutate_version[s3_cohort.report_ref["uri"]] = (
            "drifted-report-version"
        )
    elif mutation == "sealed-release-vs-manifest":
        setup.manifest.sealed_evaluation_release_sha256 = digest(
            "foreign-sealed-release"
        )
    elif mutation == "lock-body-drift":
        setup.downloads.mutate_body[s3_cohort.lock_ref["uri"]] = (
            s3_cohort.lock_bytes + b"\n"
        )
    elif mutation == "seed-receipt-body-drift":
        target = s3_cohort.seed_rows[0]["uri"]
        body, _version = s3_cohort.bodies[target]
        setup.downloads.mutate_body[target] = body + b"\n"
    elif mutation == "output-unavailable":
        from msctl.errors import MsctlError

        target = s3_cohort.index_value["outputs"][0]["members"][4]["uri"]

        original = setup.downloads.run_json

        def unavailable(argv, *, operation, timeout_seconds):
            argv = list(argv)
            bucket = argv[argv.index("--bucket") + 1]
            key = argv[argv.index("--key") + 1]
            if f"s3://{bucket}/{key}" == target:
                setup.downloads.calls.append(
                    {
                        "argv": argv,
                        "operation": operation,
                        "timeout_seconds": timeout_seconds,
                        "runner_calls_before": len(setup.runner.calls),
                    }
                )
                raise MsctlError("AWS_COMMAND_FAILED", "object missing")
            return original(
                argv,
                operation=operation,
                timeout_seconds=timeout_seconds,
            )

        setup.downloads.run_json = unavailable

    with pytest.raises(Exception) as caught:
        setup.backend.collect_cohort_evidence(
            release=setup.release,
            manifest=setup.manifest,
            cohort_report=triple,
            evidence_index=setup.index_path,
            out=out,
            apply=True,
        )

    assert getattr(caught.value, "code", None) == expected_code, mutation
    assert len(setup.downloads.calls) == expected_downloads, mutation
    if mutation != "destination-exists":
        assert not out.exists(), mutation
    assert not any(
        path.name.startswith(".") and "collect-cohort" in path.name
        for path in tmp_path.iterdir()
    ), mutation
    assert not any(
        "put-object" in argv for argv, _operation in setup.runner.calls
    ), mutation
    if mutation != "state-conflict":
        from msctl.state import StateStore

        store = StateStore(setup.backend.state_root)
        with store.locked():
            assert store.read_cohort_collection() is None


@pytest.mark.parametrize("seed", list(range(9)))
def test_collect_cohort_rejects_nonterminal_seeds_before_any_aws_call(
    tmp_path,
    monkeypatch,
    s3_cohort,
    seed,
):
    setup = _collect_setup(
        tmp_path,
        monkeypatch.setattr,
        s3_cohort,
        seed=seed,
    )
    out = tmp_path / "cohort-evidence"

    with pytest.raises(Exception) as caught:
        _collect_cohort(setup, out)

    assert getattr(caught.value, "code", None) == (
        "COHORT_COLLECT_EVIDENCE_INVALID"
    )
    assert setup.runner.calls == []
    assert setup.downloads.calls == []
    assert not out.exists()


def test_collect_cohort_rejects_instance_drift_against_the_lock(
    tmp_path,
    monkeypatch,
    s3_cohort,
):
    setup = _collect_setup(
        tmp_path,
        monkeypatch.setattr,
        s3_cohort,
        instance_id=FOREIGN_INSTANCE_ID,
    )
    out = tmp_path / "cohort-evidence"

    with pytest.raises(Exception) as caught:
        _collect_cohort(setup, out)

    assert getattr(caught.value, "code", None) == (
        "COHORT_COLLECT_EVIDENCE_INVALID"
    )
    # The report and lock were fetched; the lifecycle comparison fails
    # before any receipt or output body download.
    assert len(setup.downloads.calls) == 2
    _assert_no_mutation(setup, out)


# ---------------------------------------------------------------------------
# Independent stream rehash (brief test 6).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutation",
    ["body-drift", "length-drift", "version-drift"],
)
def test_collect_cohort_rehash_catches_artifact_drift(
    tmp_path,
    monkeypatch,
    s3_cohort,
    mutation,
):
    setup = _collect_setup(tmp_path, monkeypatch.setattr, s3_cohort)
    target = s3_cohort.index_value["outputs"][3]["members"][1]
    assert target["member"] == "items.jsonl"
    uri = target["uri"]
    body, _version = s3_cohort.bodies[uri]
    if mutation == "body-drift":
        setup.downloads.mutate_body[uri] = body[:-1] + b"X"
    elif mutation == "length-drift":
        setup.downloads.mutate_body[uri] = body + b"Y"
    elif mutation == "version-drift":
        setup.downloads.mutate_version[uri] = "drifted-version"
    out = tmp_path / "cohort-evidence"

    with pytest.raises(Exception) as caught:
        _collect_cohort(setup, out)

    assert getattr(caught.value, "code", None) == (
        "COHORT_COLLECT_OBJECT_MISMATCH"
    )
    _assert_no_mutation(setup, out)


# ---------------------------------------------------------------------------
# Dishonest-report rejection: replay is the only outcome authority
# (brief test 7).
# ---------------------------------------------------------------------------


def _republish_report(s3, report_value: dict) -> tuple[dict, dict, dict]:
    from evals.confirmatory.contracts import canonical_json_bytes

    dishonest_bytes = canonical_json_bytes(report_value)
    dishonest_sha256 = hashlib.sha256(dishonest_bytes).hexdigest()
    report_ref = {
        "uri": f"{S3_ROOT}/" + cohort_report_object_key(dishonest_sha256),
        "sha256": dishonest_sha256,
        "version_id": "cohort-report-version-1",
        "bytes": len(dishonest_bytes),
    }
    index_value = copy.deepcopy(s3.index_value)
    index_value["cohort_report"] = dict(report_ref)
    bodies = dict(s3.bodies)
    bodies[report_ref["uri"]] = (dishonest_bytes, report_ref["version_id"])
    return report_ref, index_value, bodies


def _dishonest_delta(value: dict) -> None:
    value["primary"]["paired_seed_deltas"][0]["delta"] = {
        "numerator": 9,
        "denominator": 10,
    }


def _dishonest_bound(value: dict) -> None:
    value["primary"]["bootstrap"]["ci_low"] = {
        "numerator": -1,
        "denominator": 2,
    }


def _dishonest_aulc(value: dict) -> None:
    value["aulc"][1]["raw_area"] = {"numerator": 13_582, "denominator": 1}
    value["aulc"][1]["normalized_area"] = {"numerator": 1, "denominator": 1}


def _dishonest_status(value: dict) -> None:
    value["status"]["final_inference_conclusion"] = (
        "supports_practical_null"
    )


@pytest.mark.parametrize(
    "mutate",
    [_dishonest_delta, _dishonest_bound, _dishonest_aulc, _dishonest_status],
    ids=["delta", "bound", "aulc", "status"],
)
def test_hash_consistent_dishonest_reports_fail_only_at_replay(
    tmp_path,
    monkeypatch,
    s3_cohort,
    mutate,
):
    dishonest = json.loads(s3_cohort.report_bytes)
    mutate(dishonest)
    report_ref, index_value, bodies = _republish_report(s3_cohort, dishonest)
    setup = _collect_setup(
        tmp_path,
        monkeypatch.setattr,
        s3_cohort,
        index_value=index_value,
        bodies=bodies,
    )
    out = tmp_path / "cohort-evidence"

    with pytest.raises(Exception) as caught:
        _collect_cohort(setup, out)

    assert getattr(caught.value, "code", None) == (
        "COHORT_COLLECT_REPLAY_FAILED"
    )
    # Every object downloaded cleanly; only the replay rejected the
    # dishonest outcome claims.
    assert len(setup.downloads.calls) == 1_012
    _assert_no_mutation(setup, out)


@pytest.mark.parametrize("artifact", ["outcomes.jsonl", "metrics.json"])
def test_coherently_republished_tampered_artifacts_fail_replay(
    tmp_path,
    monkeypatch,
    s3_cohort,
    artifact,
):
    from evals.confirmatory.contracts import canonical_json_bytes

    slot = s3_cohort.index_value["outputs"][0]
    output_dir = s3_cohort.cohort.references[0].output_dir
    original = (output_dir / artifact).read_bytes()
    if artifact == "outcomes.jsonl":
        lines = original.splitlines(keepends=True)
        forged_index = None
        for index, line in enumerate(lines):
            outcome = json.loads(line)
            if outcome["submitted_answer"] != "wrong-answer":
                outcome["submitted_answer"] = "forged-answer"
                lines[index] = canonical_json_bytes(outcome)
                forged_index = index
                break
        assert forged_index is not None
        tampered = b"".join(lines)
    else:
        metrics = json.loads(original)
        cells = metrics["summaries"][0]["primary_cells"]
        rate = cells[sorted(cells)[0]]
        rate["numerator"] = (
            rate["numerator"] - 1
            if rate["numerator"] > 0
            else rate["numerator"] + 1
        )
        rate["value"] = rate["numerator"] / rate["denominator"]
        tampered = canonical_json_bytes(metrics)
    assert tampered != original

    manifest_bytes = (output_dir / "output.json").read_bytes()
    manifest_value = json.loads(manifest_bytes)
    for artifact_row in manifest_value["artifacts"]:
        if artifact_row["path"] == artifact:
            artifact_row["sha256"] = hashlib.sha256(tampered).hexdigest()
            artifact_row["bytes"] = len(tampered)
    tampered_manifest = canonical_json_bytes(manifest_value)
    tampered_commitment = hashlib.sha256(tampered_manifest).hexdigest()

    report_value = json.loads(s3_cohort.report_bytes)
    report_value["inputs"][0]["output_commitment"] = tampered_commitment
    report_value["snapshots"][0]["output_commitment"] = tampered_commitment
    report_ref, index_value, bodies = _republish_report(
        s3_cohort,
        report_value,
    )

    members = index_value["outputs"][0]["members"]
    for member_name, content in (
        (artifact, tampered),
        ("output.json", tampered_manifest),
    ):
        sha256 = hashlib.sha256(content).hexdigest()
        uri = f"{S3_ROOT}/" + evaluation_output_member_key(
            slot["seed"],
            slot["arm"],
            slot["optimizer_step"],
            member_name,
            sha256,
        )
        for row in members:
            if row["member"] == member_name:
                row["sha256"] = sha256
                row["uri"] = uri
                row["bytes"] = len(content)
                bodies[uri] = (content, row["version_id"])

    setup = _collect_setup(
        tmp_path,
        monkeypatch.setattr,
        s3_cohort,
        index_value=index_value,
        bodies=bodies,
    )
    out = tmp_path / "cohort-evidence"

    with pytest.raises(Exception) as caught:
        _collect_cohort(setup, out)

    assert getattr(caught.value, "code", None) == (
        "COHORT_COLLECT_REPLAY_FAILED"
    )
    assert len(setup.downloads.calls) == 1_012
    _assert_no_mutation(setup, out)


# ---------------------------------------------------------------------------
# No-replace publication and lost-PUT recovery (brief test 8).
# ---------------------------------------------------------------------------


def test_lost_put_recovers_only_through_exact_head(
    tmp_path,
    monkeypatch,
    s3_cohort,
):
    from msctl.errors import MsctlError

    setup = _collect_setup(tmp_path, monkeypatch.setattr, s3_cohort)
    record: dict[str, object] = {}
    put, head = _publish_cohort_outputs(record)

    def lost_put(argv: list[str]):
        put(argv)
        raise MsctlError("AWS_COMMAND_FAILED", "connection dropped")

    setup.runner.outputs = [lost_put, head]
    out = tmp_path / "cohort-evidence"

    result = _collect_cohort(setup, out)

    assert result["collected"] == 1_012
    assert result["cohort_collection_receipt"]["version_id"] == (
        "cohort-collection-version-1"
    )
    assert out.exists()


def test_lost_put_with_drifted_head_conflicts_without_mutation(
    tmp_path,
    monkeypatch,
    s3_cohort,
):
    from msctl.errors import MsctlError

    setup = _collect_setup(tmp_path, monkeypatch.setattr, s3_cohort)
    record: dict[str, object] = {}
    put, _head = _publish_cohort_outputs(record)

    def lost_put(argv: list[str]):
        put(argv)
        raise MsctlError("AWS_COMMAND_FAILED", "precondition failed")

    def drifted_head(argv: list[str]):
        del argv
        return {
            "object": {
                "checksum_sha256": _checksum("9" * 64),
                "content_length": 3,
                "metadata": record["metadata"],
                "version_id": "foreign-version",
            }
        }

    setup.runner.outputs = [lost_put, drifted_head]
    out = tmp_path / "cohort-evidence"

    with pytest.raises(Exception) as caught:
        _collect_cohort(setup, out)

    assert getattr(caught.value, "code", None) == "COHORT_COLLECT_CONFLICT"
    assert not out.exists()
    from msctl.state import StateStore

    store = StateStore(setup.backend.state_root)
    with store.locked():
        assert store.read_cohort_collection() is None


def test_post_publication_head_drift_conflicts(
    tmp_path,
    monkeypatch,
    s3_cohort,
):
    setup = _collect_setup(tmp_path, monkeypatch.setattr, s3_cohort)
    record: dict[str, object] = {}
    put, _head = _publish_cohort_outputs(record)

    def version_drift_head(argv: list[str]):
        del argv
        return {
            "object": {
                "checksum_sha256": record["checksum"],
                "content_length": len(record["body"]),
                "metadata": record["metadata"],
                "version_id": "cohort-collection-version-2",
            }
        }

    setup.runner.outputs = [put, version_drift_head]
    out = tmp_path / "cohort-evidence"

    with pytest.raises(Exception) as caught:
        _collect_cohort(setup, out)

    assert getattr(caught.value, "code", None) == "COHORT_COLLECT_CONFLICT"
    assert not out.exists()


# ---------------------------------------------------------------------------
# Idempotent replay and state anchors (brief test 9).
# ---------------------------------------------------------------------------


def test_collect_cohort_replays_idempotently_and_conflicts_fail(
    tmp_path,
    monkeypatch,
    s3_cohort,
):
    setup = _collect_setup(tmp_path, monkeypatch.setattr, s3_cohort)
    record: dict[str, object] = {}
    put, head = _publish_cohort_outputs(record)
    setup.runner.outputs = [put, head]
    first = _collect_cohort(setup, tmp_path / "cohort-evidence")
    assert first["idempotent"] is False

    setup.runner.calls.clear()
    setup.downloads.calls.clear()
    reference = first["cohort_collection_receipt"]
    setup.runner.outputs = [
        {
            "object": {
                "checksum_sha256": _checksum(reference["sha256"]),
                "content_length": reference["bytes"],
                "version_id": reference["version_id"],
            }
        }
    ]
    replay = _collect_cohort(setup, tmp_path / "cohort-evidence-second")
    assert replay["idempotent"] is True
    assert replay["collected"] == 0
    assert replay["cohort_collection_receipt"] == reference
    assert setup.downloads.calls == []
    assert [operation for _argv, operation in setup.runner.calls] == [
        "verify published cohort collection receipt"
    ]
    assert not (tmp_path / "cohort-evidence-second").exists()

    setup.runner.calls.clear()
    # A conflicting republication presents a coherent anchor-plus-index
    # pair whose report identity differs from the recorded singleton.
    conflicting_sha256 = digest("conflicting-report")
    conflicting_index = copy.deepcopy(setup.index_value)
    conflicting_index["cohort_report"] = {
        "uri": f"{S3_ROOT}/" + cohort_report_object_key(conflicting_sha256),
        "sha256": conflicting_sha256,
        "version_id": "cohort-report-version-9",
        "bytes": len(s3_cohort.report_bytes),
    }
    conflicting_path = tmp_path / "conflicting-evidence-index.json"
    conflicting_path.write_bytes(
        canonical_receipt_bytes(conflicting_index)
    )
    conflicting = {
        "uri": conflicting_index["cohort_report"]["uri"],
        "sha256": conflicting_sha256,
        "version_id": "cohort-report-version-9",
    }
    with pytest.raises(Exception) as caught:
        setup.backend.collect_cohort_evidence(
            release=setup.release,
            manifest=setup.manifest,
            cohort_report=conflicting,
            evidence_index=conflicting_path,
            out=tmp_path / "cohort-evidence-third",
            apply=True,
        )
    assert getattr(caught.value, "code", None) == "COHORT_COLLECT_CONFLICT"
    assert setup.runner.calls == []
    assert setup.downloads.calls == []

    setup.runner.outputs = [
        {
            "object": {
                "checksum_sha256": _checksum("9" * 64),
                "content_length": 1,
                "version_id": "foreign-version",
            }
        }
    ]
    with pytest.raises(Exception) as caught:
        _collect_cohort(setup, tmp_path / "cohort-evidence-fourth")
    assert getattr(caught.value, "code", None) == "COHORT_COLLECT_INCOMPLETE"


# ---------------------------------------------------------------------------
# Cohort singleton state (brief test 10).
# ---------------------------------------------------------------------------


def _cohort_state(
    binding,
    *,
    study_lock: dict | None = None,
    cohort_report: dict | None = None,
    cohort_collection_receipt: dict | None = None,
    run_manifest_sha256: str | None = None,
    sealed_evaluation_release_sha256: str | None = None,
    preregistration_sha256: str | None = None,
) -> dict[str, object]:
    lock_sha256 = digest("state-study-lock")
    report_sha256 = digest("state-cohort-report")
    receipt_sha256 = digest("state-cohort-collection-receipt")
    return {
        **binding.to_dict(),
        "schema_version": 2,
        "operation": "collect-cohort",
        "run_manifest_sha256": (
            run_manifest_sha256 or digest("run-manifest-9")
        ),
        "release_sha256": RELEASE_SHA256,
        "release_receipt_sha256": RELEASE_RECEIPT_SHA256,
        "sealed_evaluation_release_sha256": (
            sealed_evaluation_release_sha256 or digest("sealed-release")
        ),
        "preregistration_sha256": (
            preregistration_sha256 or digest("preregistration")
        ),
        "study_lock": study_lock
        or {
            "uri": f"{S3_ROOT}/" + study_lock_object_key(lock_sha256),
            "sha256": lock_sha256,
            "version_id": "study-lock-version-1",
            "bytes": 8_192,
        },
        "cohort_report": cohort_report
        or {
            "uri": f"{S3_ROOT}/" + cohort_report_object_key(report_sha256),
            "sha256": report_sha256,
            "version_id": "cohort-report-version-1",
            "bytes": 16_384,
        },
        "cohort_collection_receipt": cohort_collection_receipt
        or {
            "uri": (
                f"{S3_ROOT}/" + cohort_collection_receipt_key(receipt_sha256)
            ),
            "sha256": receipt_sha256,
            "version_id": "cohort-collection-version-1",
            "bytes": 262_144,
        },
        "objects_collected": 1_012,
        "bytes_collected": 123_456,
        "status": "Published",
        "created_at": "2026-07-24T04:00:00Z",
        "updated_at": "2026-07-24T04:00:00Z",
    }


def test_cohort_state_roundtrip_and_updated_at_only_rewrite(tmp_path):
    from msctl.errors import MsctlError
    from msctl.state import StateStore

    binding = _terminal_binding()
    state = _cohort_state(binding)
    store = StateStore(tmp_path / "state")

    with pytest.raises(MsctlError) as caught:
        store.read_cohort_collection()
    assert caught.value.code == "UNSAFE_STATE"

    with store.locked():
        assert store.read_cohort_collection() is None
        store.write_cohort_collection(state)
        assert store.read_cohort_collection() == state

        replay = dict(state)
        replay["updated_at"] = "2026-07-24T05:00:00Z"
        store.write_cohort_collection(replay)
        assert store.read_cohort_collection() == replay

        conflicting = dict(replay)
        conflicting["bytes_collected"] = 999
        with pytest.raises(MsctlError) as caught:
            store.write_cohort_collection(conflicting)
        assert caught.value.code == "STATE_CORRUPT"
        assert store.read_cohort_collection() == replay

    cohorts = tmp_path / "state" / "cohorts"
    assert cohorts.is_dir()
    files = sorted(path.name for path in cohorts.iterdir())
    assert len(files) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "extra-key",
        "missing-key",
        "schema-version",
        "operation",
        "provider",
        "status",
        "seed-not-terminal",
        "objects-collected",
        "bytes-collected-zero",
        "bytes-collected-bool",
        "lifecycle-binding",
        "study-lock-key",
        "report-key",
        "receipt-key",
        "receipt-shape",
        "sealed-release-shape",
        "preregistration-shape",
    ],
)
def test_cohort_state_mutation_matrix_fails_closed(tmp_path, mutation):
    from msctl.errors import MsctlError
    from msctl.state import StateStore

    binding = _terminal_binding()
    state = _cohort_state(binding)
    if mutation == "extra-key":
        state["out"] = "/tmp/evidence"
    elif mutation == "missing-key":
        state.pop("bytes_collected")
    elif mutation == "schema-version":
        state["schema_version"] = 1
    elif mutation == "operation":
        state["operation"] = "collect"
    elif mutation == "provider":
        state["provider"] = "illumina-usfc-prd"
    elif mutation == "status":
        state["status"] = "Pending"
    elif mutation == "seed-not-terminal":
        state["seed"] = 8
    elif mutation == "objects-collected":
        state["objects_collected"] = 1_011
    elif mutation == "bytes-collected-zero":
        state["bytes_collected"] = 0
    elif mutation == "bytes-collected-bool":
        state["bytes_collected"] = True
    elif mutation == "lifecycle-binding":
        state["purchase_model"] = ""
    elif mutation == "study-lock-key":
        state["study_lock"]["uri"] = (
            f"{S3_ROOT}/"
            + cohort_report_object_key(state["study_lock"]["sha256"])
        )
    elif mutation == "report-key":
        state["cohort_report"]["uri"] = (
            f"{S3_ROOT}/"
            + study_lock_object_key(state["cohort_report"]["sha256"])
        )
    elif mutation == "receipt-key":
        state["cohort_collection_receipt"]["uri"] = (
            f"{S3_ROOT}/"
            + cohort_collection_receipt_key("c" * 64)
        )
    elif mutation == "receipt-shape":
        state["cohort_collection_receipt"].pop("bytes")
    elif mutation == "sealed-release-shape":
        state["sealed_evaluation_release_sha256"] = "X" * 64
    elif mutation == "preregistration-shape":
        state["preregistration_sha256"] = "not-a-hash"
    store = StateStore(tmp_path / "state")

    with store.locked(), pytest.raises(MsctlError) as caught:
        store.write_cohort_collection(state)

    assert caught.value.code in {"STATE_CORRUPT", "SCHEMA_INVALID"}, mutation
    with store.locked():
        assert store.read_cohort_collection() is None


def test_cohort_state_singleton_makes_second_cohort_unrepresentable(tmp_path):
    from msctl.errors import MsctlError
    from msctl.state import StateStore

    binding = _terminal_binding()
    store = StateStore(tmp_path / "state")
    first = _cohort_state(binding)
    second = _cohort_state(
        binding,
        cohort_report={
            "uri": (
                f"{S3_ROOT}/"
                + cohort_report_object_key(digest("second-report"))
            ),
            "sha256": digest("second-report"),
            "version_id": "cohort-report-version-9",
            "bytes": 32_768,
        },
    )

    with store.locked():
        store.write_cohort_collection(first)
        with pytest.raises(MsctlError) as caught:
            store.write_cohort_collection(second)
        assert caught.value.code == "STATE_CORRUPT"
        assert store.read_cohort_collection() == first
    cohorts = tmp_path / "state" / "cohorts"
    assert len(list(cohorts.iterdir())) == 1


# ---------------------------------------------------------------------------
# Evaluate and cleanup stay blocked (brief test 12).
# ---------------------------------------------------------------------------


def test_evaluate_and_cleanup_stay_blocked_with_full_collection_chain(
    tmp_path,
    monkeypatch,
    s3_cohort,
):
    from tests.test_aws_collect import _collection_state
    from tests.provider_lifecycle_fixtures import provider_lifecycle
    from msctl.state import StateStore

    setup = _collect_setup(tmp_path, monkeypatch.setattr, s3_cohort)
    backend = setup.backend
    profile = load_aws_gpu_profile(P5_PROFILE)
    store = StateStore(backend.state_root)
    for seed in range(10):
        binding = provider_lifecycle(profile, seed=seed).binding
        manifest_sha256 = hashlib.sha256(
            f"collected-manifest-{seed}".encode("ascii")
        ).hexdigest()
        with store.locked():
            store.write_collection(
                manifest_sha256,
                _collection_state(
                    binding,
                    manifest_sha256=manifest_sha256,
                ),
            )
    with store.locked():
        store.write_cohort_collection(
            _cohort_state(
                setup.binding,
                study_lock=dict(s3_cohort.lock_ref),
                cohort_report=dict(s3_cohort.report_ref),
                run_manifest_sha256=setup.manifest.sha256,
                sealed_evaluation_release_sha256=(
                    s3_cohort.sealed_release_sha256
                ),
                preregistration_sha256=s3_cohort.preregistration_sha256,
            )
        )
    monkeypatch.setattr(
        backend,
        "_load_bound_inputs",
        lambda **_kwargs: (setup.release, setup.manifest),
    )
    arguments = SimpleNamespace(
        apply=True,
        release=tmp_path / "RELEASE.json",
        manifest=tmp_path / "manifest.json",
        approval=tmp_path / "approval.json",
        repo_root=tmp_path,
        cached=False,
    )

    for command in ("evaluate", "cleanup plan", "cleanup apply"):
        with pytest.raises(Exception) as caught:
            backend.dispatch(command, arguments)
        assert getattr(caught.value, "code", None) == (
            "OPERATION_UNSUPPORTED"
        ), command
        message = str(caught.value)
        if command == "evaluate":
            assert "Task 4E" in message
        else:
            assert "ten per-seed collection receipts" in message
            assert "cohort evaluation-evidence collection receipt" in message
            assert "Task 4E" in message
    assert not any(
        "terminate-instances" in argv
        for argv, _operation in setup.runner.calls
    )
    assert setup.runner.calls == []
    assert setup.downloads.calls == []


# ---------------------------------------------------------------------------
# Selected/legacy dispatch separation (brief test 13).
# ---------------------------------------------------------------------------


def test_collect_cohort_dispatch_requires_complete_argument_set(
    tmp_path,
    monkeypatch,
    s3_cohort,
):
    setup = _collect_setup(tmp_path, monkeypatch.setattr, s3_cohort)
    backend = setup.backend
    monkeypatch.setattr(
        backend,
        "_load_bound_inputs",
        lambda **_kwargs: (setup.release, setup.manifest),
    )

    with pytest.raises(Exception) as caught:
        backend.dispatch(
            "collect-cohort",
            SimpleNamespace(
                apply=False,
                release=tmp_path / "release.json",
                manifest=tmp_path / "manifest.json",
                cohort_report_uri=setup.anchor["uri"],
                cohort_report_sha256=None,
                cohort_report_version_id=setup.anchor["version_id"],
                evidence_index=None,
                out=None,
                repo_root=tmp_path,
            ),
        )
    assert getattr(caught.value, "code", None) == "CLI_USAGE"
    missing = caught.value.details["missing"]
    assert "--cohort-report-sha256" in missing
    assert "--evidence-index" in missing
    assert "--out" in missing

    dry_run, planned = backend.dispatch(
        "collect-cohort",
        SimpleNamespace(
            apply=False,
            release=tmp_path / "release.json",
            manifest=tmp_path / "manifest.json",
            cohort_report_uri=setup.anchor["uri"],
            cohort_report_sha256=setup.anchor["sha256"],
            cohort_report_version_id=setup.anchor["version_id"],
            evidence_index=setup.index_path,
            out=tmp_path / "cohort-evidence",
            repo_root=tmp_path,
        ),
    )
    assert dry_run is True
    assert planned["collected"] == 0
    assert planned["provider"] == "aws-p5.48xlarge"
    assert setup.runner.calls == []
    assert setup.downloads.calls == []


# ---------------------------------------------------------------------------
# Import boundary (brief test 17).
# ---------------------------------------------------------------------------


def _forbidden_modules(*names: str) -> str:
    return (
        "banned = sorted(\n"
        "    name for name in sys.modules\n"
        f"    if any(name == b or name.startswith(b + '.') for b in {names!r})\n"
        ")\n"
        "assert not banned, banned\n"
    )


def test_import_and_dry_run_leave_confirmatory_unimported(tmp_path):
    # Importing the new module alone must pull neither torch, nor
    # train.trainer, nor evals.confirmatory. The controller module itself
    # has always pulled torch through the frozen Task 3D finalization
    # import chain, so the dry-run half asserts the strongest attainable
    # boundary: evals.confirmatory stays unimported throughout, and the
    # dry run itself introduces no torch module beyond that frozen chain.
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import msctl.aws_cohort_collect\n"
        + _forbidden_modules("torch", "train.trainer", "evals.confirmatory")
        + "from pathlib import Path\n"
        "import tests.test_aws_cohort_collect as fixture\n"
        + _forbidden_modules("evals.confirmatory")
        + "tmp = Path(sys.argv[1])\n"
        "backend, lifecycle, runner = fixture._cohort_backend(\n"
        "    tmp,\n"
        "    lambda module, name, value: setattr(module, name, value),\n"
        ")\n"
        "manifest = fixture._cohort_manifest(\n"
        "    lifecycle.binding,\n"
        "    sealed_release_sha256=fixture.digest('sealed-release'),\n"
        "    preregistration_sha256=fixture.digest('preregistration'),\n"
        ")\n"
        "release = fixture._cohort_release(manifest)\n"
        "index_value = fixture._synthetic_index()\n"
        "index_path = tmp / 'cohort-evidence-index.json'\n"
        "index_path.write_bytes(\n"
        "    fixture.canonical_receipt_bytes(index_value)\n"
        ")\n"
        "anchor = {\n"
        "    'uri': index_value['cohort_report']['uri'],\n"
        "    'sha256': index_value['cohort_report']['sha256'],\n"
        "    'version_id': index_value['cohort_report']['version_id'],\n"
        "}\n"
        "before = {name for name in sys.modules if name == 'torch'\n"
        "          or name.startswith('torch.')}\n"
        "result = backend.collect_cohort_evidence(\n"
        "    release=release,\n"
        "    manifest=manifest,\n"
        "    cohort_report=anchor,\n"
        "    evidence_index=index_path,\n"
        "    out=tmp / 'cohort-evidence',\n"
        "    apply=False,\n"
        ")\n"
        "assert result['collected'] == 0\n"
        "assert runner.calls == []\n"
        "after = {name for name in sys.modules if name == 'torch'\n"
        "         or name.startswith('torch.')}\n"
        "assert after == before, sorted(after - before)[:5]\n"
        + _forbidden_modules("evals.confirmatory")
        + "print('IMPORT-BOUNDARY-OK')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert completed.returncode == 0, completed.stderr
    assert "IMPORT-BOUNDARY-OK" in completed.stdout


def _script_apply(tmp: Path) -> tuple[dict[str, object], list[str]]:
    """Full apply used by the subprocess torch-boundary proof."""

    from tests.cohort_output_fixtures import _build_cohort

    cohort_root = tmp / "cohort"
    cohort_root.mkdir(mode=0o700)
    cohort = _build_cohort(cohort_root, "effect")
    s3 = _wrap_cohort(cohort)
    setup = _collect_setup(
        tmp,
        lambda module, name, value: setattr(module, name, value),
        s3,
    )
    record: dict[str, object] = {}
    put, head = _publish_cohort_outputs(record)
    setup.runner.outputs = [put, head]
    before = {
        name
        for name in sys.modules
        if name == "torch" or name.startswith("torch.")
    }
    result = _collect_cohort(setup, tmp / "cohort-evidence")
    added = sorted(
        name
        for name in sys.modules
        if (name == "torch" or name.startswith("torch."))
        and name not in before
    )
    return result, added


def test_apply_replay_imports_confirmatory_locally_but_never_torch(
    tmp_path,
):
    # The apply-path replay imports evals.confirmatory.aggregate
    # function-locally; the replay path is snapshot-free and must not
    # introduce a single torch module.
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\n"
        "import tests.test_aws_cohort_collect as fixture\n"
        "result, torch_added = fixture._script_apply(Path(sys.argv[1]))\n"
        "assert result['collected'] == 1_012, result\n"
        "assert 'evals.confirmatory.aggregate' in sys.modules\n"
        "assert torch_added == [], torch_added[:5]\n"
        "print('APPLY-BOUNDARY-OK')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert completed.returncode == 0, completed.stderr
    assert "APPLY-BOUNDARY-OK" in completed.stdout
