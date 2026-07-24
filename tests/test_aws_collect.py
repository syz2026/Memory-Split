from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cluster.aws.gpu_profile import load_aws_gpu_profile
from msctl.aws_contracts import SNAPSHOT_STEPS
from tests.provider_lifecycle_fixtures import provider_lifecycle
from tests.test_aws_seed_transition import (
    P5_PROFILE,
    P6_PROFILE,
    S3_ROOT,
    _admit_arguments,
    _canonical,
    _receipt_ref,
    _receipt_value,
)


COLLECTION_RECEIPT_TYPE = "memorysplit-aws-seed-collection-v3"


def _collection_value(
    run_value: dict[str, object],
    run_payload: bytes,
    run_ref,
    *,
    checkpoint_receipt_bytes: int = 1_000,
) -> dict[str, object]:
    seed = run_value["seed"]
    objects: list[dict[str, object]] = []
    for arm_row in run_value["arms"]:
        arm = arm_row["arm"]
        for item in arm_row["snapshots"]:
            snapshot = item["object"]
            objects.append(
                {
                    "arm": arm,
                    "bytes": snapshot["bytes"],
                    "kind": "snapshot",
                    "sha256": snapshot["sha256"],
                    "step": item["step"],
                    "uri": snapshot["uri"],
                    "version_id": snapshot["version_id"],
                }
            )
        log = arm_row["log"]
        objects.append(
            {
                "arm": arm,
                "bytes": log["bytes"],
                "kind": "log",
                "sha256": log["sha256"],
                "step": None,
                "uri": log["uri"],
                "version_id": log["version_id"],
            }
        )
    for arm in ("dense", "split90"):
        digest = hashlib.sha256(
            f"checkpoint-object-{seed}-{arm}".encode("ascii")
        ).hexdigest()
        objects.append(
            {
                "arm": arm,
                "bytes": 8_192,
                "kind": "checkpoint",
                "sha256": digest,
                "step": None,
                "uri": (
                    f"{S3_ROOT}/checkpoints/seed-{seed}/{arm}/"
                    f"sha256/{digest}.pt"
                ),
                "version_id": f"checkpoint-object-{arm}-1",
            }
        )
    checkpoint = run_value["checkpoint_receipt"]
    objects.append(
        {
            "arm": None,
            "bytes": checkpoint_receipt_bytes,
            "kind": "checkpoint_receipt",
            "sha256": checkpoint["sha256"],
            "step": None,
            "uri": checkpoint["uri"],
            "version_id": checkpoint["version_id"],
        }
    )
    objects.append(
        {
            "arm": None,
            "bytes": len(run_payload),
            "kind": "run_receipt",
            "sha256": run_ref.sha256,
            "step": None,
            "uri": run_ref.uri,
            "version_id": run_ref.version_id,
        }
    )
    value = {
        field: copy.deepcopy(run_value[field])
        for field in run_value
        if field
        not in {
            "arms",
            "checkpoint_receipt",
            "finalized_at",
            "receipt_type",
            "request_id",
        }
    }
    value.update(
        {
            "receipt_type": COLLECTION_RECEIPT_TYPE,
            "request_id": "b" * 32,
            "collected_at": "2026-07-24T02:00:00Z",
            "run_receipt": {
                "bytes": len(run_payload),
                "sha256": run_ref.sha256,
                "uri": run_ref.uri,
                "version_id": run_ref.version_id,
            },
            "checkpoint_receipt": {
                "bytes": checkpoint_receipt_bytes,
                "sha256": checkpoint["sha256"],
                "uri": checkpoint["uri"],
                "version_id": checkpoint["version_id"],
            },
            "objects": objects,
        }
    )
    return value


def _collection_ref(
    value: dict[str, object],
    *,
    payload: bytes | None = None,
):
    from msctl.aws_collect import CollectionReceiptRef

    body = payload if payload is not None else _canonical(value)
    digest = hashlib.sha256(body).hexdigest()
    return body, CollectionReceiptRef(
        uri=(
            f"{S3_ROOT}/receipts/collections/seed-{value['seed']}/"
            f"sha256/{digest}.json"
        ),
        sha256=digest,
        version_id="collection-receipt-version-1",
    )


def _prior_collection_fixture(*, profile_path=P5_PROFILE, seed: int = 1):
    """One admitted prior finalization plus its matching collection value."""

    from msctl.aws_seed_transition import admit_prior_seed_finalization

    profile = load_aws_gpu_profile(profile_path)
    binding = provider_lifecycle(profile, seed=seed).binding
    run_value = _receipt_value(binding, seed=seed - 1)
    run_payload, run_ref = _receipt_ref(run_value)
    admitted = admit_prior_seed_finalization(
        run_payload,
        ref=run_ref,
        **_admit_arguments(binding),
    )
    collection = _collection_value(run_value, run_payload, run_ref)
    return SimpleNamespace(
        binding=binding,
        run_value=run_value,
        run_payload=run_payload,
        run_ref=run_ref,
        admitted=admitted,
        collection=collection,
    )


def test_collection_contracts_are_exact():
    from msctl import aws_collect

    assert aws_collect.COLLECTION_RECEIPT_TYPE == COLLECTION_RECEIPT_TYPE
    assert aws_collect.SEED_COLLECTION_OBJECT_COUNT == 16
    assert type(aws_collect.SEED_COLLECTION_OBJECT_COUNT) is int
    assert issubclass(aws_collect.CollectionError, ValueError)


def test_collection_fields_mirror_portable_run_finalization_contract():
    from cluster.aws.p5.run_finalization import _RECEIPT_FIELDS
    from msctl.aws_collect import COLLECTION_RECEIPT_FIELDS

    assert COLLECTION_RECEIPT_FIELDS == frozenset(
        (_RECEIPT_FIELDS - {"arms", "finalized_at"})
        | {"collected_at", "objects", "run_receipt"}
    )


@pytest.mark.parametrize("profile_path", [P5_PROFILE, P6_PROFILE])
def test_parse_seed_collection_receipt_roundtrip(profile_path):
    from msctl.aws_collect import parse_seed_collection_receipt_bytes

    profile = load_aws_gpu_profile(profile_path)
    binding = provider_lifecycle(profile, seed=2).binding
    run_value = _receipt_value(binding, seed=2)
    run_payload, run_ref = _receipt_ref(run_value)
    value = _collection_value(run_value, run_payload, run_ref)
    payload, ref = _collection_ref(value)

    parsed = parse_seed_collection_receipt_bytes(
        payload,
        receipt_uri=ref.uri,
        receipt_sha256=ref.sha256,
        receipt_version_id=ref.version_id,
        expected_binding=replace(binding, seed=2),
    )

    assert parsed == value
    rows = parsed["objects"]
    assert len(rows) == 16
    assert [row["kind"] for row in rows] == [
        *(["snapshot"] * 5),
        "log",
        *(["snapshot"] * 5),
        "log",
        "checkpoint",
        "checkpoint",
        "checkpoint_receipt",
        "run_receipt",
    ]
    assert [row["arm"] for row in rows] == [
        *(["dense"] * 6),
        *(["split90"] * 6),
        "dense",
        "split90",
        None,
        None,
    ]
    assert [row["step"] for row in rows[:5]] == list(SNAPSHOT_STEPS)
    assert rows[14] == {
        "arm": None,
        "kind": "checkpoint_receipt",
        "step": None,
        **value["checkpoint_receipt"],
    }
    assert rows[15] == {
        "arm": None,
        "kind": "run_receipt",
        "step": None,
        **value["run_receipt"],
    }


_TOP_LEVEL_MUTATIONS = (
    "boot_id",
    "canary_receipt_sha256",
    "cohort_id",
    "collected_at",
    "complete",
    "dataset_build_id",
    "dataset_receipt_sha256",
    "environment_receipt_sha256",
    "hardware_amendment_sha256",
    "instance_id",
    "objective_controls_contract_sha256",
    "ordered_stream_sha256",
    "profile_id",
    "profile_sha256",
    "provider",
    "provider_selection_sha256",
    "provider_selection_version_id",
    "qualification_approval_public_key_sha256",
    "qualification_approval_receipt_sha256",
    "qualification_evidence_sha256",
    "receipt_type",
    "release_receipt_sha256",
    "release_sha256",
    "request_id",
    "run_manifest_sha256",
    "runtime_lock_sha256",
    "runtime_sbom_sha256",
    "schema_version",
    "seed",
    "source_commit",
    "source_tree",
)


@pytest.mark.parametrize("field", _TOP_LEVEL_MUTATIONS)
def test_parser_top_level_mutation_matrix_fails(field):
    from msctl.aws_collect import (
        CollectionError,
        parse_seed_collection_receipt_bytes,
    )

    profile = load_aws_gpu_profile(P6_PROFILE)
    binding = provider_lifecycle(profile, seed=4).binding
    run_value = _receipt_value(binding, seed=4)
    run_payload, run_ref = _receipt_ref(run_value)
    value = _collection_value(run_value, run_payload, run_ref)
    current = value[field]
    if field == "seed":
        value[field] = 5
        for row in value["objects"]:
            row["uri"] = str(row["uri"]).replace("seed-4", "seed-5")
        for reference in ("run_receipt", "checkpoint_receipt"):
            value[reference]["uri"] = str(
                value[reference]["uri"]
            ).replace("seed-4", "seed-5")
        value["objects"][15]["uri"] = value["run_receipt"]["uri"]
        value["objects"][14]["uri"] = value["checkpoint_receipt"]["uri"]
    elif field == "complete":
        value[field] = False
    elif field == "schema_version":
        value[field] = 2
    elif field == "collected_at":
        value[field] = "2026-07-24 02:00:00"
    elif field == "request_id":
        value[field] = "not-a-request"
    elif field in {"provider", "profile_id"}:
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
    elif field == "instance_id":
        value[field] = "i-0fedcba9876543210"
    elif field == "boot_id":
        value[field] = "87654321-4321-4cba-9abc-0987654321fe"
    elif field in {"source_commit", "source_tree"}:
        # Provenance fields are shape-validated by the parser; their exact
        # values are cross-bound by the collection controller against the
        # authenticated manifest and finalization receipt.
        value[field] = "x" * 40
    elif field in {
        "dataset_build_id",
        "dataset_receipt_sha256",
        "ordered_stream_sha256",
        "release_receipt_sha256",
        "release_sha256",
        "run_manifest_sha256",
    }:
        value[field] = "X" * 64
    elif field == "receipt_type":
        value[field] = "memorysplit-aws-paired-run-finalization-v3"
    elif isinstance(current, str) and len(current) == 64:
        value[field] = ("0" if current[0] != "0" else "1") * 64
    else:
        value[field] = "mutated-" + str(current)
    payload, ref = _collection_ref(value)

    with pytest.raises(CollectionError):
        parse_seed_collection_receipt_bytes(
            payload,
            receipt_uri=ref.uri,
            receipt_sha256=ref.sha256,
            receipt_version_id=ref.version_id,
            expected_binding=replace(binding, seed=4),
        )


def test_parser_rejects_unknown_or_missing_fields():
    from msctl.aws_collect import (
        CollectionError,
        parse_seed_collection_receipt_bytes,
    )

    profile = load_aws_gpu_profile(P5_PROFILE)
    binding = provider_lifecycle(profile, seed=1).binding
    run_value = _receipt_value(binding, seed=1)
    run_payload, run_ref = _receipt_ref(run_value)
    for mutation in ("extra", "missing"):
        value = _collection_value(run_value, run_payload, run_ref)
        if mutation == "extra":
            value["finalized_at"] = "2026-07-24T00:00:00Z"
        else:
            value.pop("collected_at")
        payload, ref = _collection_ref(value)
        with pytest.raises(CollectionError):
            parse_seed_collection_receipt_bytes(
                payload,
                receipt_uri=ref.uri,
                receipt_sha256=ref.sha256,
                receipt_version_id=ref.version_id,
            )


@pytest.mark.parametrize(
    "mutation",
    [
        "appended-byte",
        "hash-mismatch",
        "null-version",
        "empty-version",
        "uri-wrong-seed",
        "uri-wrong-digest",
        "uri-unsafe-root",
        "uri-not-s3",
        "duplicate-key",
        "non-finite",
        "not-canonical-order",
        "empty-payload",
    ],
)
def test_parser_rejects_noncanonical_bytes_or_identity(mutation):
    from msctl.aws_collect import (
        CollectionError,
        parse_seed_collection_receipt_bytes,
    )

    profile = load_aws_gpu_profile(P5_PROFILE)
    binding = provider_lifecycle(profile, seed=3).binding
    run_value = _receipt_value(binding, seed=3)
    run_payload, run_ref = _receipt_ref(run_value)
    value = _collection_value(run_value, run_payload, run_ref)
    payload, ref = _collection_ref(value)
    receipt_uri = ref.uri
    receipt_sha256 = ref.sha256
    receipt_version_id = ref.version_id
    if mutation == "appended-byte":
        payload = payload + b"\n"
    elif mutation == "hash-mismatch":
        receipt_sha256 = "c" * 64
    elif mutation == "null-version":
        receipt_version_id = "null"
    elif mutation == "empty-version":
        receipt_version_id = ""
    elif mutation == "uri-wrong-seed":
        receipt_uri = receipt_uri.replace("seed-3", "seed-4")
    elif mutation == "uri-wrong-digest":
        receipt_uri = (
            f"{S3_ROOT}/receipts/collections/seed-3/sha256/"
            + "c" * 64
            + ".json"
        )
    elif mutation == "uri-unsafe-root":
        receipt_uri = (
            "s3://memorysplit-prod/../confirmatory-v3/receipts/"
            f"collections/seed-3/sha256/{receipt_sha256}.json"
        )
    elif mutation == "uri-not-s3":
        receipt_uri = (
            "https://memorysplit-prod/confirmatory-v3/receipts/"
            f"collections/seed-3/sha256/{receipt_sha256}.json"
        )
    elif mutation == "duplicate-key":
        payload = (
            b'{"seed":3,"seed":3'
            + payload[1 + payload.index(b","):]
        )
        receipt_sha256 = hashlib.sha256(payload).hexdigest()
    elif mutation == "non-finite":
        payload = payload.replace(b'"bytes":2048', b'"bytes":NaN', 1)
        receipt_sha256 = hashlib.sha256(payload).hexdigest()
    elif mutation == "not-canonical-order":
        decoded = json.loads(payload.decode("ascii"))
        reordered = dict(reversed(list(decoded.items())))
        payload = json.dumps(
            reordered,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii") + b"\n"
        receipt_sha256 = hashlib.sha256(payload).hexdigest()
    elif mutation == "empty-payload":
        payload = b""
        receipt_sha256 = hashlib.sha256(payload).hexdigest()

    with pytest.raises(CollectionError):
        parse_seed_collection_receipt_bytes(
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
        "reordered-rows",
        "duplicate-row",
        "cross-receipt-row",
        "foreign-kind",
        "snapshot-arm-none",
        "snapshot-step-off-schedule",
        "log-with-step",
        "bytes-zero",
        "bytes-bool",
        "row-version-null",
        "row-uri-key-mismatch",
        "row-fields-extra",
        "row-fields-missing",
        "checkpoint-row-differs-from-reference",
        "run-row-differs-from-reference",
        "objects-not-a-list",
    ],
)
def test_object_rows_are_exact(mutation):
    from msctl.aws_collect import (
        CollectionError,
        parse_seed_collection_receipt_bytes,
    )

    profile = load_aws_gpu_profile(P5_PROFILE)
    binding = provider_lifecycle(profile, seed=2).binding
    run_value = _receipt_value(binding, seed=2)
    run_payload, run_ref = _receipt_ref(run_value)
    value = _collection_value(run_value, run_payload, run_ref)
    rows = value["objects"]
    if mutation == "missing-row":
        rows.pop(3)
    elif mutation == "extra-row":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "reordered-rows":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "duplicate-row":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "cross-receipt-row":
        foreign_value = _receipt_value(
            provider_lifecycle(profile, seed=3).binding,
            seed=3,
        )
        foreign_payload, foreign_ref = _receipt_ref(foreign_value)
        foreign = _collection_value(
            foreign_value,
            foreign_payload,
            foreign_ref,
        )
        rows[0] = foreign["objects"][0]
    elif mutation == "foreign-kind":
        rows[0]["kind"] = "artifact"
    elif mutation == "snapshot-arm-none":
        rows[0]["arm"] = None
    elif mutation == "snapshot-step-off-schedule":
        rows[0]["step"] = 1_359
    elif mutation == "log-with-step":
        rows[5]["step"] = 13_582
    elif mutation == "bytes-zero":
        rows[0]["bytes"] = 0
    elif mutation == "bytes-bool":
        rows[0]["bytes"] = True
    elif mutation == "row-version-null":
        rows[0]["version_id"] = "null"
    elif mutation == "row-uri-key-mismatch":
        rows[0]["uri"] = rows[0]["uri"].replace("step-1358", "step-3396")
    elif mutation == "row-fields-extra":
        rows[0]["etag"] = "abc"
    elif mutation == "row-fields-missing":
        rows[0].pop("version_id")
    elif mutation == "checkpoint-row-differs-from-reference":
        rows[14]["bytes"] = rows[14]["bytes"] + 1
    elif mutation == "run-row-differs-from-reference":
        rows[15]["version_id"] = "prior-receipt-version-2"
    elif mutation == "objects-not-a-list":
        value["objects"] = {"rows": rows}
    payload, ref = _collection_ref(value)

    with pytest.raises(CollectionError):
        parse_seed_collection_receipt_bytes(
            payload,
            receipt_uri=ref.uri,
            receipt_sha256=ref.sha256,
            receipt_version_id=ref.version_id,
        )


@pytest.mark.parametrize("profile_path", [P5_PROFILE, P6_PROFILE])
def test_admit_prior_seed_collection_returns_two_checkpoints(profile_path):
    from msctl.aws_collect import (
        AdmittedPriorCollection,
        CollectedObject,
        admit_prior_seed_collection,
    )

    fixture = _prior_collection_fixture(profile_path=profile_path, seed=3)
    payload, ref = _collection_ref(fixture.collection)

    admitted = admit_prior_seed_collection(
        payload,
        ref=ref,
        admitted=fixture.admitted,
        binding=fixture.binding,
    )

    assert isinstance(admitted, AdmittedPriorCollection)
    assert admitted.seed == 2
    assert admitted.receipt == ref
    assert len(admitted.checkpoints) == 2
    rows = fixture.collection["objects"]
    for checkpoint, row in zip(admitted.checkpoints, rows[12:14]):
        assert isinstance(checkpoint, CollectedObject)
        assert checkpoint.kind == "checkpoint"
        assert checkpoint.arm == row["arm"]
        assert checkpoint.step is None
        assert checkpoint.uri == row["uri"]
        assert checkpoint.sha256 == row["sha256"]
        assert checkpoint.bytes == row["bytes"]
        assert checkpoint.version_id == row["version_id"]
    assert [checkpoint.arm for checkpoint in admitted.checkpoints] == [
        "dense",
        "split90",
    ]


def test_admit_seed_zero_collection_is_forbidden():
    from msctl.aws_seed_transition import admit_prior_seed_finalization
    from msctl.aws_collect import (
        CollectionError,
        admit_prior_seed_collection,
    )

    profile = load_aws_gpu_profile(P5_PROFILE)
    binding = provider_lifecycle(profile, seed=1).binding
    run_value = _receipt_value(binding, seed=0)
    run_payload, run_ref = _receipt_ref(run_value)
    admitted = admit_prior_seed_finalization(
        run_payload,
        ref=run_ref,
        **_admit_arguments(binding),
    )
    collection = _collection_value(run_value, run_payload, run_ref)
    payload, ref = _collection_ref(collection)

    with pytest.raises(CollectionError, match="[Ss]eed 0"):
        admit_prior_seed_collection(
            payload,
            ref=ref,
            admitted=admitted,
            binding=provider_lifecycle(profile, seed=0).binding,
        )


@pytest.mark.parametrize("receipt_seed", [3, 1, 0])
def test_admit_wrong_collection_seed_fails(receipt_seed):
    from msctl.aws_collect import (
        CollectionError,
        admit_prior_seed_collection,
    )

    fixture = _prior_collection_fixture(seed=3)
    foreign_binding = provider_lifecycle(
        load_aws_gpu_profile(P5_PROFILE),
        seed=receipt_seed,
    ).binding
    foreign_run = _receipt_value(foreign_binding, seed=receipt_seed)
    foreign_payload, foreign_ref = _receipt_ref(foreign_run)
    collection = _collection_value(
        foreign_run,
        foreign_payload,
        foreign_ref,
    )
    payload, ref = _collection_ref(collection)

    with pytest.raises(CollectionError, match="seed"):
        admit_prior_seed_collection(
            payload,
            ref=ref,
            admitted=fixture.admitted,
            binding=fixture.binding,
        )


def test_admit_boot_only_drift_passes_after_reboot():
    from msctl.aws_seed_transition import admit_prior_seed_finalization
    from msctl.aws_collect import admit_prior_seed_collection

    profile = load_aws_gpu_profile(P5_PROFILE)
    binding = provider_lifecycle(profile, seed=1).binding
    run_value = _receipt_value(binding, seed=0)
    run_value["boot_id"] = "87654321-4321-4cba-9abc-0987654321fe"
    run_payload, run_ref = _receipt_ref(run_value)
    admitted = admit_prior_seed_finalization(
        run_payload,
        ref=run_ref,
        **_admit_arguments(binding),
    )
    collection = _collection_value(run_value, run_payload, run_ref)
    payload, ref = _collection_ref(collection)

    admitted_collection = admit_prior_seed_collection(
        payload,
        ref=ref,
        admitted=admitted,
        binding=binding,
    )

    assert admitted_collection.seed == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "instance",
        "runtime-lock",
        "run-receipt-triple",
        "run-receipt-bytes",
        "snapshot-row",
        "log-row",
        "checkpoint-receipt-row",
    ],
)
def test_admit_authority_instance_and_row_drift_fails(mutation):
    from msctl.aws_collect import (
        CollectionError,
        admit_prior_seed_collection,
    )

    fixture = _prior_collection_fixture(seed=2)
    collection = fixture.collection
    if mutation == "instance":
        collection["instance_id"] = "i-0fedcba9876543210"
    elif mutation == "runtime-lock":
        collection["runtime_lock_sha256"] = "f" * 64
    elif mutation == "run-receipt-triple":
        # A collection receipt for different run-receipt bytes parses but
        # must not admit against this AdmittedPriorRun.
        foreign_value = copy.deepcopy(fixture.run_value)
        foreign_value["request_id"] = "8" * 32
        foreign_payload, foreign_ref = _receipt_ref(foreign_value)
        collection = _collection_value(
            foreign_value,
            foreign_payload,
            foreign_ref,
        )
    elif mutation == "run-receipt-bytes":
        collection["run_receipt"]["bytes"] = (
            collection["run_receipt"]["bytes"] + 1
        )
        collection["objects"][15]["bytes"] = (
            collection["run_receipt"]["bytes"]
        )
    elif mutation == "snapshot-row":
        drifted = "e" * 64
        row = collection["objects"][0]
        row["sha256"] = drifted
        row["uri"] = (
            f"{S3_ROOT}/snapshots/seed-1/dense/step-1358/"
            f"sha256/{drifted}.pt"
        )
    elif mutation == "log-row":
        collection["objects"][5]["bytes"] = 4_096
    elif mutation == "checkpoint-receipt-row":
        collection["objects"][14]["version_id"] = "checkpoint-receipt-2"
        collection["checkpoint_receipt"]["version_id"] = (
            "checkpoint-receipt-2"
        )
    payload, ref = _collection_ref(collection)

    with pytest.raises(CollectionError):
        admit_prior_seed_collection(
            payload,
            ref=ref,
            admitted=fixture.admitted,
            binding=fixture.binding,
        )


def test_collection_dataclasses_fail_closed():
    from msctl.aws_collect import (
        AdmittedPriorCollection,
        CollectedObject,
        CollectionError,
        CollectionReceiptRef,
    )

    with pytest.raises(CollectionError):
        CollectionReceiptRef(
            uri="https://x/y.json",
            sha256="a" * 64,
            version_id="v1",
        )
    with pytest.raises(CollectionError):
        CollectionReceiptRef(
            uri="s3://bucket/key.json",
            sha256="A" * 64,
            version_id="v1",
        )
    with pytest.raises(CollectionError):
        CollectionReceiptRef(
            uri="s3://bucket/key.json",
            sha256="a" * 64,
            version_id="null",
        )
    with pytest.raises(CollectionError):
        CollectedObject(
            kind="artifact",
            arm=None,
            step=None,
            uri="s3://bucket/object.pt",
            sha256="a" * 64,
            bytes=1,
            version_id="v1",
        )
    with pytest.raises(CollectionError):
        CollectedObject(
            kind="snapshot",
            arm="dense",
            step=999,
            uri="s3://bucket/object.pt",
            sha256="a" * 64,
            bytes=1,
            version_id="v1",
        )
    with pytest.raises(CollectionError):
        CollectedObject(
            kind="checkpoint",
            arm="dense",
            step=None,
            uri="s3://bucket/object.pt",
            sha256="a" * 64,
            bytes=0,
            version_id="v1",
        )
    checkpoints = tuple(
        CollectedObject(
            kind="checkpoint",
            arm=arm,
            step=None,
            uri=f"s3://bucket/{arm}.pt",
            sha256="a" * 64,
            bytes=1,
            version_id="v1",
        )
        for arm in ("dense", "split90")
    )
    ref = CollectionReceiptRef(
        uri="s3://bucket/key.json",
        sha256="a" * 64,
        version_id="v1",
    )
    with pytest.raises(CollectionError, match="seed"):
        AdmittedPriorCollection(seed=9, receipt=ref, checkpoints=checkpoints)
    with pytest.raises(CollectionError):
        AdmittedPriorCollection(
            seed=0,
            receipt=ref,
            checkpoints=(checkpoints[0], checkpoints[0]),
        )
    admitted = AdmittedPriorCollection(
        seed=0,
        receipt=ref,
        checkpoints=checkpoints,
    )
    assert admitted.seed == 0


def test_download_timeout_is_bounded_by_expected_bytes():
    from msctl.aws_collect import (
        CollectionError,
        download_timeout_seconds,
    )

    assert download_timeout_seconds(1) == 300.0
    assert download_timeout_seconds(1_024) == 300.0
    assert download_timeout_seconds(8 * 1024 * 1024 * 1024) == 2_048.0
    assert download_timeout_seconds(2 * 1024 * 1024 * 1024) == 512.0
    for invalid in (0, -1, True, 1.5, "10", None):
        with pytest.raises(CollectionError):
            download_timeout_seconds(invalid)


def test_download_runner_uses_explicit_timeout_and_empty_environment(
    monkeypatch,
):
    from msctl.aws_collect import SubprocessAwsDownloadRunner

    observed = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    value = SubprocessAwsDownloadRunner().run_json(
        ["env", "-i", "aws", "s3api", "get-object"],
        operation="collect snapshot",
        timeout_seconds=1_234.5,
    )

    assert value == {"ok": True}
    assert observed["kwargs"]["timeout"] == 1_234.5
    assert observed["kwargs"]["env"] == {}
    assert observed["kwargs"]["shell"] is False

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=3,
            stdout="secret-value",
            stderr="secret-value",
        ),
    )
    with pytest.raises(Exception) as caught:
        SubprocessAwsDownloadRunner().run_json(
            ["env", "-i", "aws", "s3api", "get-object"],
            operation="collect snapshot",
            timeout_seconds=300.0,
        )
    assert getattr(caught.value, "code", None) == "AWS_COMMAND_FAILED"
    assert "secret-value" not in str(caught.value)

    def timeout_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout_run)
    with pytest.raises(Exception) as caught:
        SubprocessAwsDownloadRunner().run_json(
            ["env", "-i", "aws", "s3api", "get-object"],
            operation="collect snapshot",
            timeout_seconds=300.0,
        )
    assert getattr(caught.value, "code", None) == "EXTERNAL_UNAVAILABLE"

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"a":1,"a":2}',
            stderr="",
        ),
    )
    with pytest.raises(Exception) as caught:
        SubprocessAwsDownloadRunner().run_json(
            ["env", "-i", "aws", "s3api", "get-object"],
            operation="collect snapshot",
            timeout_seconds=300.0,
        )
    assert getattr(caught.value, "code", None) == "AWS_OUTPUT_INVALID"


RELEASE_SHA256 = "d" * 64
RELEASE_RECEIPT_SHA256 = "e" * 64


class _ScriptedDownloadRunner:
    """Serve exact bodies by URI and record ordering and timeouts."""

    def __init__(self, bodies: dict[str, tuple[bytes, str]], runner) -> None:
        self.bodies = bodies
        self.runner = runner
        self.calls: list[dict[str, object]] = []
        self.mutate_body: dict[str, bytes] = {}
        self.mutate_version: dict[str, str] = {}

    def run_json(self, argv, *, operation: str, timeout_seconds: float):
        import base64

        argv = list(argv)
        self.calls.append(
            {
                "argv": argv,
                "operation": operation,
                "timeout_seconds": timeout_seconds,
                "runner_calls_before": len(self.runner.calls),
            }
        )
        bucket = argv[argv.index("--bucket") + 1]
        key = argv[argv.index("--key") + 1]
        uri = f"s3://{bucket}/{key}"
        destination = argv[argv.index("--checksum-mode") + 2]
        body, version_id = self.bodies[uri]
        written = self.mutate_body.get(uri, body)
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(written)
        return {
            "object": {
                "checksum_sha256": base64.b64encode(
                    hashlib.sha256(body).digest()
                ).decode("ascii"),
                "content_length": len(body),
                "version_id": self.mutate_version.get(uri, version_id),
            }
        }


def _paired_checkpoint_receipt_value(
    manifest,
    *,
    boot_id: str | None = None,
    step: int = 13_582,
) -> tuple[dict[str, object], dict[str, tuple[bytes, str]]]:
    from msctl.aws_lifecycle import LIFECYCLE_BINDING_FIELDS

    bodies: dict[str, tuple[bytes, str]] = {}
    rows = []
    for run in manifest.runs:
        body = (
            f"final-checkpoint-body-{manifest.seed}-{run.arm}".encode("ascii")
            * 7
        )
        digest = hashlib.sha256(body).hexdigest()
        uri = (
            f"{S3_ROOT}/checkpoints/seed-{manifest.seed}/{run.arm}/"
            f"sha256/{digest}.pt"
        )
        bodies[uri] = (body, f"final-{run.arm}-1")
        rows.append(
            {
                "arm": run.arm,
                "checkpoint_version": 3,
                "config_fingerprint": "3" * 64,
                "config_sha256": run.config_sha256,
                "data": {
                    "build_id": manifest.dataset_build_id,
                    "global_cursor": step * 524_288,
                    "ordered_stream_sha256": (
                        manifest.ordered_stream_sha256
                    ),
                    "receipt_sha256": manifest.dataset_receipt_sha256,
                    "sidecar_name": (
                        "dense_target_weights"
                        if run.arm == "dense"
                        else "split90_target_weights"
                    ),
                },
                "object": {
                    "bytes": len(body),
                    "sha256": digest,
                    "uri": uri,
                    "version_id": f"final-{run.arm}-1",
                },
                "run_id": run.run_id,
                "seed": manifest.seed,
                "step": step,
                "world_size": 4,
            }
        )
    value = {
        **{
            field: (
                list(manifest.arms)
                if field == "arms"
                else getattr(manifest, field)
            )
            for field in LIFECYCLE_BINDING_FIELDS
        },
        "boot_id": boot_id or manifest.boot_id,
        "checkpoints": rows,
        "cohort_id": manifest.cohort_id,
        "dataset_build_id": manifest.dataset_build_id,
        "dataset_receipt_sha256": manifest.dataset_receipt_sha256,
        "environment_receipt_sha256": "8" * 64,
        "freshness": {
            "deadline_at": "2026-07-24T01:20:00Z",
            "max_age_seconds": 1_200,
            "requested_at": "2026-07-24T01:00:00Z",
            "staged_at": "2026-07-24T01:05:00Z",
        },
        "instance_id": manifest.instance_id,
        "ordered_stream_sha256": manifest.ordered_stream_sha256,
        "profile_sha256": manifest.profile_sha256,
        "provider": manifest.provider,
        "reason": "periodic",
        "receipt_type": "memorysplit-aws-paired-checkpoint-v3",
        "release_receipt_sha256": manifest.release_receipt_sha256,
        "release_sha256": manifest.release_sha256,
        "request_id": "c" * 32,
        "resumable": True,
        "run_manifest_sha256": manifest.sha256,
        "schema_version": 3,
        "seed": manifest.seed,
        "source_commit": manifest.source_commit,
        "source_tree": manifest.source_tree,
    }
    return value, bodies


def _collect_fixture(tmp_path, monkeypatch, *, seed: int = 1, profile_path=None):
    from tests.test_aws_seed_transition import (
        _selected_backend,
        _selected_flow_manifest,
        _selected_release,
    )

    backend, lifecycle, runner = _selected_backend(
        tmp_path,
        monkeypatch,
        seed=seed,
        **(
            {"profile_path": profile_path}
            if profile_path is not None
            else {}
        ),
    )
    manifest = _selected_flow_manifest(lifecycle.binding)
    release = _selected_release(manifest)
    binding = lifecycle.binding

    checkpoint_value, bodies = _paired_checkpoint_receipt_value(manifest)
    checkpoint_payload = _canonical(checkpoint_value)
    checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
    checkpoint_uri = (
        f"{S3_ROOT}/receipts/checkpoints/seed-{seed}/"
        f"sha256/{checkpoint_sha256}.json"
    )

    run_value = _receipt_value(
        binding,
        seed=seed,
        run_manifest_sha256=manifest.sha256,
    )
    run_value["checkpoint_receipt"] = {
        "sha256": checkpoint_sha256,
        "uri": checkpoint_uri,
        "version_id": "final-checkpoint-receipt-1",
    }
    by_arm = {run.arm: run for run in manifest.runs}
    for arm_row in run_value["arms"]:
        arm = arm_row["arm"]
        arm_row["config_sha256"] = by_arm[arm].config_sha256
        for item in arm_row["snapshots"]:
            body = (
                f"snapshot-body-{seed}-{arm}-{item['step']}".encode("ascii")
                * 5
            )
            digest = hashlib.sha256(body).hexdigest()
            item["object"]["sha256"] = digest
            item["object"]["bytes"] = len(body)
            item["object"]["uri"] = (
                f"{S3_ROOT}/snapshots/seed-{seed}/{arm}/"
                f"step-{item['step']}/sha256/{digest}.pt"
            )
            bodies[item["object"]["uri"]] = (
                body,
                item["object"]["version_id"],
            )
        # Training logs are opaque evidence: a body that is not even valid
        # UTF-8 JSON must still collect when its hash matches.
        log_body = b'{"broken": json \xff\x00 not utf8'
        log_digest = hashlib.sha256(log_body).hexdigest()
        arm_row["log"]["sha256"] = log_digest
        arm_row["log"]["bytes"] = len(log_body)
        arm_row["log"]["uri"] = (
            f"{S3_ROOT}/logs/seed-{seed}/{arm}/sha256/{log_digest}.jsonl"
        )
        bodies[arm_row["log"]["uri"]] = (
            log_body,
            arm_row["log"]["version_id"],
        )
    run_payload, run_ref = _receipt_ref(run_value)

    downloads = _ScriptedDownloadRunner(bodies, runner)
    backend.download_runner = downloads
    return SimpleNamespace(
        backend=backend,
        binding=binding,
        bodies=bodies,
        checkpoint_payload=checkpoint_payload,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_uri=checkpoint_uri,
        checkpoint_value=checkpoint_value,
        downloads=downloads,
        manifest=manifest,
        release=release,
        run_payload=run_payload,
        run_ref=run_ref,
        run_value=run_value,
        runner=runner,
    )


def _run_triple(fixture) -> dict[str, str]:
    return {
        "uri": fixture.run_ref.uri,
        "sha256": fixture.run_ref.sha256,
        "version_id": "final-run-receipt-1",
    }


def _receipt_get_output(payload: bytes, version_id: str):
    from tests.test_aws_seed_transition import _download

    import base64

    return _download(
        payload,
        {
            "receipt": {
                "checksum_sha256": base64.b64encode(
                    hashlib.sha256(payload).digest()
                ).decode("ascii"),
                "version_id": version_id,
            }
        },
    )


def _publish_outputs(record: dict[str, object]):
    import base64

    def _put(argv: list[str]):
        argv = list(argv)
        record["put_argv"] = argv
        checksum = argv[argv.index("--checksum-sha256") + 1]
        record["checksum"] = checksum
        record["metadata"] = dict(
            part.split("=", 1)
            for part in argv[argv.index("--metadata") + 1].split(",")
        )
        record["body"] = Path(
            argv[argv.index("--body") + 1]
        ).read_bytes()
        return {
            "object": {
                "checksum_sha256": checksum,
                "version_id": "collection-version-1",
            }
        }

    def _head(argv: list[str]):
        del argv
        return {
            "object": {
                "checksum_sha256": record["checksum"],
                "content_length": len(record["body"]),
                "metadata": record["metadata"],
                "version_id": "collection-version-1",
            }
        }

    return _put, _head


def test_collect_seed_evidence_dry_run_renders_first_get_only(
    tmp_path,
    monkeypatch,
):
    for profile_path, provider in (
        (P5_PROFILE, "aws-p5.48xlarge"),
        (P6_PROFILE, "aws-p6-b300.48xlarge"),
    ):
        fixture = _collect_fixture(
            tmp_path / provider,
            monkeypatch,
            seed=1,
            profile_path=profile_path,
        )
        out = tmp_path / provider / "evidence"

        result = fixture.backend.collect_seed_evidence(
            release=fixture.release,
            manifest=fixture.manifest,
            run_receipt=_run_triple(fixture),
            out=out,
            apply=False,
        )

        assert result["provider"] == provider
        assert result["seed"] == 1
        assert result["collected"] == 0
        assert result["idempotent"] is False
        assert result["out"] == str(out)
        assert len(result["commands"]) == 1
        first = result["commands"][0]
        assert "get-object" in first
        assert fixture.run_ref.sha256 in " ".join(first)
        assert "--version-id" in first
        assert fixture.runner.calls == []
        assert fixture.downloads.calls == []
        assert not out.exists()


def test_collect_seed_evidence_full_flow_is_ordered_and_durable(
    tmp_path,
    monkeypatch,
):
    from msctl.aws_collect import (
        download_timeout_seconds,
        parse_seed_collection_receipt_bytes,
    )
    from msctl.state import StateStore

    fixture = _collect_fixture(tmp_path, monkeypatch, seed=1)
    out = tmp_path / "evidence"
    publish_record: dict[str, object] = {}
    put, head = _publish_outputs(publish_record)
    fixture.runner.outputs = [
        _receipt_get_output(fixture.run_payload, "final-run-receipt-1"),
        _receipt_get_output(
            fixture.checkpoint_payload,
            "final-checkpoint-receipt-1",
        ),
        put,
        head,
    ]

    result = fixture.backend.collect_seed_evidence(
        release=fixture.release,
        manifest=fixture.manifest,
        run_receipt=_run_triple(fixture),
        out=out,
        apply=True,
    )

    assert result["provider"] == "aws-p5.48xlarge"
    assert result["seed"] == 1
    assert result["run_manifest_sha256"] == fixture.manifest.sha256
    assert result["objects_collected"] == 16
    assert result["collected"] == 16
    assert result["idempotent"] is False
    assert result["out"] == str(out)
    expected_bytes = sum(
        len(body) for body, _version in fixture.bodies.values()
    ) + len(fixture.run_payload) + len(fixture.checkpoint_payload)
    assert result["bytes_collected"] == expected_bytes

    operations = [operation for _argv, operation in fixture.runner.calls]
    assert operations == [
        "fetch seed run finalization receipt",
        "fetch seed checkpoint receipt",
        "publish seed collection receipt",
        "verify seed collection receipt",
    ]
    assert len(fixture.downloads.calls) == 14
    assert all(
        call["runner_calls_before"] == 2 for call in fixture.downloads.calls
    )
    for call in fixture.downloads.calls:
        argv = call["argv"]
        bucket = argv[argv.index("--bucket") + 1]
        key = argv[argv.index("--key") + 1]
        body, _version = fixture.bodies[f"s3://{bucket}/{key}"]
        assert call["timeout_seconds"] == download_timeout_seconds(
            len(body)
        )
        assert "--version-id" in argv

    receipt_bytes = publish_record["body"]
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    assert result["collection_receipt"] == {
        "bytes": len(receipt_bytes),
        "sha256": receipt_sha256,
        "uri": (
            f"{S3_ROOT}/receipts/collections/seed-1/"
            f"sha256/{receipt_sha256}.json"
        ),
        "version_id": "collection-version-1",
    }
    assert "--if-none-match" in publish_record["put_argv"]
    assert publish_record["metadata"]["receipt-type"] == (
        COLLECTION_RECEIPT_TYPE
    )
    parsed = parse_seed_collection_receipt_bytes(
        receipt_bytes,
        receipt_uri=result["collection_receipt"]["uri"],
        receipt_sha256=receipt_sha256,
        receipt_version_id="collection-version-1",
    )
    assert parsed["run_receipt"]["bytes"] == len(fixture.run_payload)

    prefix = f"{S3_ROOT}/"
    for row in parsed["objects"]:
        local = out / str(row["uri"]).removeprefix(prefix)
        assert local.is_file(), row["uri"]
        if row["kind"] in {"checkpoint_receipt", "run_receipt"}:
            expected = (
                fixture.checkpoint_payload
                if row["kind"] == "checkpoint_receipt"
                else fixture.run_payload
            )
            assert local.read_bytes() == expected
        else:
            body, _version = fixture.bodies[row["uri"]]
            assert local.read_bytes() == body
    receipt_local = out / "receipts" / "collections" / "seed-1" / (
        "sha256"
    ) / f"{receipt_sha256}.json"
    assert receipt_local.read_bytes() == receipt_bytes
    assert not any(
        path.name.startswith(".") and "collect" in path.name
        for path in out.parent.iterdir()
    )

    store = StateStore(fixture.backend.state_root)
    with store.locked():
        state = store.read_collection(fixture.manifest.sha256)
    assert state is not None
    assert state["operation"] == "collect"
    assert state["status"] == "Published"
    assert state["objects_collected"] == 16
    assert state["bytes_collected"] == expected_bytes
    assert state["run_receipt"] == {
        "bytes": len(fixture.run_payload),
        "sha256": fixture.run_ref.sha256,
        "uri": fixture.run_ref.uri,
        "version_id": "final-run-receipt-1",
    }
    assert state["collection_receipt"] == result["collection_receipt"]


def test_collect_replays_idempotently_and_conflicts_fail(
    tmp_path,
    monkeypatch,
):
    fixture = _collect_fixture(tmp_path, monkeypatch, seed=1)
    publish_record: dict[str, object] = {}
    put, head = _publish_outputs(publish_record)
    fixture.runner.outputs = [
        _receipt_get_output(fixture.run_payload, "final-run-receipt-1"),
        _receipt_get_output(
            fixture.checkpoint_payload,
            "final-checkpoint-receipt-1",
        ),
        put,
        head,
    ]
    first = fixture.backend.collect_seed_evidence(
        release=fixture.release,
        manifest=fixture.manifest,
        run_receipt=_run_triple(fixture),
        out=tmp_path / "evidence",
        apply=True,
    )
    assert first["idempotent"] is False

    fixture.runner.calls.clear()
    fixture.downloads.calls.clear()
    import base64

    fixture.runner.outputs = [
        {
            "object": {
                "checksum_sha256": base64.b64encode(
                    bytes.fromhex(
                        first["collection_receipt"]["sha256"]
                    )
                ).decode("ascii"),
                "content_length": first["collection_receipt"]["bytes"],
                "version_id": first["collection_receipt"]["version_id"],
            }
        }
    ]
    replay = fixture.backend.collect_seed_evidence(
        release=fixture.release,
        manifest=fixture.manifest,
        run_receipt=_run_triple(fixture),
        out=tmp_path / "evidence-second",
        apply=True,
    )
    assert replay["idempotent"] is True
    assert replay["collected"] == 0
    assert replay["collection_receipt"] == first["collection_receipt"]
    assert fixture.downloads.calls == []
    assert [
        operation for _argv, operation in fixture.runner.calls
    ] == ["verify published seed collection receipt"]
    assert not (tmp_path / "evidence-second").exists()

    fixture.runner.calls.clear()
    conflicting = dict(_run_triple(fixture))
    conflicting["sha256"] = "f" * 64
    conflicting["uri"] = (
        f"{S3_ROOT}/receipts/runs/seed-1/sha256/" + "f" * 64 + ".json"
    )
    with pytest.raises(Exception) as caught:
        fixture.backend.collect_seed_evidence(
            release=fixture.release,
            manifest=fixture.manifest,
            run_receipt=conflicting,
            out=tmp_path / "evidence-third",
            apply=True,
        )
    assert getattr(caught.value, "code", None) == "COLLECT_CONFLICT"
    assert fixture.runner.calls == []

    fixture.runner.calls.clear()
    fixture.runner.outputs = [
        {
            "object": {
                "checksum_sha256": base64.b64encode(
                    bytes.fromhex("9" * 64)
                ).decode("ascii"),
                "content_length": 1,
                "version_id": "foreign-version",
            }
        }
    ]
    with pytest.raises(Exception) as caught:
        fixture.backend.collect_seed_evidence(
            release=fixture.release,
            manifest=fixture.manifest,
            run_receipt=_run_triple(fixture),
            out=tmp_path / "evidence-fourth",
            apply=True,
        )
    assert getattr(caught.value, "code", None) == "COLLECT_INCOMPLETE"


def test_collect_lost_put_recovers_only_through_exact_head(
    tmp_path,
    monkeypatch,
):
    from msctl.errors import MsctlError

    fixture = _collect_fixture(tmp_path, monkeypatch, seed=1)
    publish_record: dict[str, object] = {}
    put, head = _publish_outputs(publish_record)

    def _lost_put_response(argv: list[str]):
        put(argv)
        raise MsctlError("AWS_COMMAND_FAILED", "connection dropped")

    fixture.runner.outputs = [
        _receipt_get_output(fixture.run_payload, "final-run-receipt-1"),
        _receipt_get_output(
            fixture.checkpoint_payload,
            "final-checkpoint-receipt-1",
        ),
        _lost_put_response,
        head,
    ]
    result = fixture.backend.collect_seed_evidence(
        release=fixture.release,
        manifest=fixture.manifest,
        run_receipt=_run_triple(fixture),
        out=tmp_path / "evidence",
        apply=True,
    )
    assert result["objects_collected"] == 16
    assert result["collection_receipt"]["version_id"] == (
        "collection-version-1"
    )

    foreign = _collect_fixture(tmp_path / "foreign", monkeypatch, seed=1)
    foreign_record: dict[str, object] = {}
    foreign_put, _foreign_head = _publish_outputs(foreign_record)

    def _foreign_lost_put(argv: list[str]):
        foreign_put(argv)
        raise MsctlError("AWS_COMMAND_FAILED", "precondition failed")

    def _drifted_head(argv: list[str]):
        del argv
        import base64

        return {
            "object": {
                "checksum_sha256": base64.b64encode(
                    bytes.fromhex("9" * 64)
                ).decode("ascii"),
                "content_length": 3,
                "metadata": foreign_record["metadata"],
                "version_id": "foreign-version",
            }
        }

    foreign.runner.outputs = [
        _receipt_get_output(foreign.run_payload, "final-run-receipt-1"),
        _receipt_get_output(
            foreign.checkpoint_payload,
            "final-checkpoint-receipt-1",
        ),
        _foreign_lost_put,
        _drifted_head,
    ]
    with pytest.raises(Exception) as caught:
        foreign.backend.collect_seed_evidence(
            release=foreign.release,
            manifest=foreign.manifest,
            run_receipt=_run_triple(foreign),
            out=tmp_path / "foreign-evidence",
            apply=True,
        )
    assert getattr(caught.value, "code", None) == "COLLECT_CONFLICT"
    assert not (tmp_path / "foreign-evidence").exists()
    from msctl.state import StateStore

    store = StateStore(foreign.backend.state_root)
    with store.locked():
        assert store.read_collection(foreign.manifest.sha256) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "body-drift",
        "version-drift",
        "length-drift",
    ],
)
def test_collect_independent_rehash_catches_body_or_version_drift(
    tmp_path,
    monkeypatch,
    mutation,
):
    fixture = _collect_fixture(tmp_path, monkeypatch, seed=1)
    target = fixture.run_value["arms"][0]["snapshots"][2]["object"]["uri"]
    if mutation == "body-drift":
        body, _version = fixture.bodies[target]
        fixture.downloads.mutate_body[target] = body[:-1] + b"X"
    elif mutation == "version-drift":
        fixture.downloads.mutate_version[target] = "drifted-version"
    elif mutation == "length-drift":
        body, _version = fixture.bodies[target]
        fixture.downloads.mutate_body[target] = body + b"Y"
    fixture.runner.outputs = [
        _receipt_get_output(fixture.run_payload, "final-run-receipt-1"),
        _receipt_get_output(
            fixture.checkpoint_payload,
            "final-checkpoint-receipt-1",
        ),
    ]
    out = tmp_path / "evidence"

    with pytest.raises(Exception) as caught:
        fixture.backend.collect_seed_evidence(
            release=fixture.release,
            manifest=fixture.manifest,
            run_receipt=_run_triple(fixture),
            out=out,
            apply=True,
        )

    assert getattr(caught.value, "code", None) == "COLLECT_OBJECT_MISMATCH"
    assert not out.exists()
    assert not any(
        "collect" in path.name for path in out.parent.iterdir()
        if path.name.startswith(".")
    )
    assert not any(
        "put-object" in argv for argv, _operation in fixture.runner.calls
    )
    from msctl.state import StateStore

    store = StateStore(fixture.backend.state_root)
    with store.locked():
        assert store.read_collection(fixture.manifest.sha256) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-triple",
        "malformed-sha",
        "foreign-root",
        "wrong-key",
        "destination-exists",
        "receipt-checksum-drift",
        "receipt-manifest-drift",
        "checkpoint-receipt-drift",
    ],
)
def test_collect_gate_failures_precede_every_mutation(
    tmp_path,
    monkeypatch,
    mutation,
):
    fixture = _collect_fixture(tmp_path, monkeypatch, seed=1)
    out = tmp_path / "evidence"
    triple = _run_triple(fixture)
    expected_code = "COLLECT_RECEIPT_INVALID"
    if mutation == "missing-triple":
        triple = None
    elif mutation == "malformed-sha":
        triple["sha256"] = "not-a-hash"
    elif mutation == "foreign-root":
        triple["uri"] = (
            "s3://foreign-bucket/receipts/runs/seed-1/"
            f"sha256/{triple['sha256']}.json"
        )
    elif mutation == "wrong-key":
        triple["uri"] = (
            f"{S3_ROOT}/receipts/runs/seed-2/"
            f"sha256/{triple['sha256']}.json"
        )
    elif mutation == "destination-exists":
        out.mkdir(parents=True)
        expected_code = "COLLECT_DESTINATION_EXISTS"
    elif mutation == "receipt-checksum-drift":
        fixture.runner.outputs = [
            _receipt_get_output(
                fixture.run_payload + b"\n",
                "final-run-receipt-1",
            ),
        ]
    elif mutation == "receipt-manifest-drift":
        # A shape-valid finalization receipt for a different run manifest
        # must fail the manifest cross-binding, not collect.
        foreign_value = copy.deepcopy(fixture.run_value)
        foreign_value["run_manifest_sha256"] = "9" * 64
        foreign_payload, foreign_ref = _receipt_ref(foreign_value)
        triple = {
            "uri": foreign_ref.uri,
            "sha256": foreign_ref.sha256,
            "version_id": "final-run-receipt-1",
        }
        fixture.runner.outputs = [
            _receipt_get_output(foreign_payload, "final-run-receipt-1"),
        ]
    elif mutation == "checkpoint-receipt-drift":
        fixture.runner.outputs = [
            _receipt_get_output(
                fixture.run_payload,
                "final-run-receipt-1",
            ),
            _receipt_get_output(
                fixture.checkpoint_payload,
                "drifted-checkpoint-version",
            ),
        ]

    with pytest.raises(Exception) as caught:
        fixture.backend.collect_seed_evidence(
            release=fixture.release,
            manifest=fixture.manifest,
            run_receipt=triple,
            out=out,
            apply=True,
        )

    assert getattr(caught.value, "code", None) == expected_code, mutation
    assert fixture.downloads.calls == [], mutation
    assert not any(
        "put-object" in argv for argv, _operation in fixture.runner.calls
    ), mutation
    if mutation != "destination-exists":
        assert not out.exists(), mutation
    assert not any(
        path.name.startswith(".") and "collect" in path.name
        for path in tmp_path.iterdir()
    ), mutation
    from msctl.state import StateStore

    store = StateStore(fixture.backend.state_root)
    with store.locked():
        assert store.read_collection(fixture.manifest.sha256) is None


def test_selected_cleanup_and_evaluate_stay_blocked_with_ten_collections(
    tmp_path,
    monkeypatch,
):
    from msctl.state import StateStore

    fixture = _collect_fixture(tmp_path, monkeypatch, seed=9)
    backend = fixture.backend
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
    monkeypatch.setattr(
        backend,
        "_load_bound_inputs",
        lambda **_kwargs: (fixture.release, fixture.manifest),
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
        if command.startswith("cleanup"):
            message = str(caught.value)
            assert "ten per-seed collection receipts" in message
            assert "sealed-evaluation" in message
    assert not any(
        "terminate-instances" in argv
        for argv, _operation in fixture.runner.calls
    )
    assert fixture.runner.calls == []


def test_collect_dispatch_routes_selected_and_legacy_separately(
    tmp_path,
    monkeypatch,
):
    fixture = _collect_fixture(tmp_path, monkeypatch, seed=1)
    backend = fixture.backend
    monkeypatch.setattr(
        backend,
        "_load_bound_inputs",
        lambda **_kwargs: (fixture.release, fixture.manifest),
    )
    triple = _run_triple(fixture)

    with pytest.raises(Exception) as caught:
        backend.dispatch(
            "collect",
            SimpleNamespace(
                apply=False,
                source="results/seed-1.json",
                out=tmp_path / "evidence",
                release=tmp_path / "release.json",
                manifest=tmp_path / "manifest.json",
                run_receipt_uri=triple["uri"],
                run_receipt_sha256=triple["sha256"],
                run_receipt_version_id=triple["version_id"],
                repo_root=tmp_path,
            ),
        )
    assert getattr(caught.value, "code", None) == "CLI_USAGE"

    with pytest.raises(Exception) as caught:
        backend.dispatch(
            "collect",
            SimpleNamespace(
                apply=False,
                source=None,
                out=tmp_path / "evidence",
                release=tmp_path / "release.json",
                manifest=tmp_path / "manifest.json",
                run_receipt_uri=triple["uri"],
                run_receipt_sha256=None,
                run_receipt_version_id=triple["version_id"],
                repo_root=tmp_path,
            ),
        )
    assert getattr(caught.value, "code", None) == "CLI_USAGE"

    dry_run, planned = backend.dispatch(
        "collect",
        SimpleNamespace(
            apply=False,
            source=None,
            out=tmp_path / "evidence",
            release=tmp_path / "release.json",
            manifest=tmp_path / "manifest.json",
            run_receipt_uri=triple["uri"],
            run_receipt_sha256=triple["sha256"],
            run_receipt_version_id=triple["version_id"],
            repo_root=tmp_path,
        ),
    )
    assert dry_run is True
    assert planned["collected"] == 0
    assert planned["provider"] == "aws-p5.48xlarge"

    with pytest.raises(Exception) as caught:
        backend.dispatch(
            "collect",
            SimpleNamespace(
                apply=False,
                source=None,
                out=tmp_path / "evidence",
            ),
        )
    assert getattr(caught.value, "code", None) == "CLI_USAGE"


def _collection_state(
    binding,
    *,
    manifest_sha256: str,
    run_receipt: dict[str, object] | None = None,
    collection_receipt: dict[str, object] | None = None,
) -> dict[str, object]:
    seed = binding.seed
    run_digest = hashlib.sha256(
        f"collected-run-{seed}".encode("ascii")
    ).hexdigest()
    collection_digest = hashlib.sha256(
        f"collection-receipt-{seed}".encode("ascii")
    ).hexdigest()
    return {
        **binding.to_dict(),
        "schema_version": 2,
        "operation": "collect",
        "run_manifest_sha256": manifest_sha256,
        "release_sha256": RELEASE_SHA256,
        "release_receipt_sha256": RELEASE_RECEIPT_SHA256,
        "run_receipt": run_receipt
        or {
            "bytes": 5_120,
            "sha256": run_digest,
            "uri": (
                f"{S3_ROOT}/receipts/runs/seed-{seed}/"
                f"sha256/{run_digest}.json"
            ),
            "version_id": "run-receipt-version-1",
        },
        "collection_receipt": collection_receipt
        or {
            "bytes": 9_216,
            "sha256": collection_digest,
            "uri": (
                f"{S3_ROOT}/receipts/collections/seed-{seed}/"
                f"sha256/{collection_digest}.json"
            ),
            "version_id": "collection-receipt-version-1",
        },
        "objects_collected": 16,
        "bytes_collected": 123_456,
        "status": "Published",
        "created_at": "2026-07-24T02:00:00Z",
        "updated_at": "2026-07-24T02:00:00Z",
    }


def test_collection_state_roundtrip_and_updated_at_only_rewrite(tmp_path):
    from msctl.errors import MsctlError
    from msctl.state import StateStore

    profile = load_aws_gpu_profile(P5_PROFILE)
    binding = provider_lifecycle(profile, seed=1).binding
    manifest_sha256 = hashlib.sha256(b"collect-manifest-1").hexdigest()
    state = _collection_state(binding, manifest_sha256=manifest_sha256)
    store = StateStore(tmp_path / "state")

    with pytest.raises(MsctlError) as caught:
        store.read_collection(manifest_sha256)
    assert caught.value.code == "UNSAFE_STATE"

    with store.locked():
        assert store.read_collection(manifest_sha256) is None
        store.write_collection(manifest_sha256, state)
        assert store.read_collection(manifest_sha256) == state

        replay = dict(state)
        replay["updated_at"] = "2026-07-24T03:00:00Z"
        store.write_collection(manifest_sha256, replay)
        assert store.read_collection(manifest_sha256) == replay

        conflicting = dict(replay)
        conflicting["bytes_collected"] = 999
        with pytest.raises(MsctlError) as caught:
            store.write_collection(manifest_sha256, conflicting)
        assert caught.value.code == "STATE_CORRUPT"
        assert store.read_collection(manifest_sha256) == replay

    assert (
        tmp_path
        / "state"
        / "collections"
        / f"{manifest_sha256}.json"
    ).is_file()


@pytest.mark.parametrize(
    "mutation",
    [
        "extra-key",
        "missing-key",
        "schema-version",
        "operation",
        "provider",
        "status",
        "objects-collected",
        "bytes-collected-zero",
        "bytes-collected-bool",
        "lifecycle-binding",
        "seed-key-mismatch",
        "run-receipt-shape",
        "run-receipt-key",
        "collection-receipt-shape",
        "collection-receipt-key",
        "manifest-binding",
    ],
)
def test_collection_state_mutation_matrix_fails_closed(tmp_path, mutation):
    from msctl.errors import MsctlError
    from msctl.state import StateStore

    profile = load_aws_gpu_profile(P6_PROFILE)
    binding = provider_lifecycle(profile, seed=2).binding
    manifest_sha256 = hashlib.sha256(b"collect-manifest-2").hexdigest()
    state = _collection_state(binding, manifest_sha256=manifest_sha256)
    written_manifest = manifest_sha256
    if mutation == "extra-key":
        state["out"] = "/tmp/evidence"
    elif mutation == "missing-key":
        state.pop("bytes_collected")
    elif mutation == "schema-version":
        state["schema_version"] = 1
    elif mutation == "operation":
        state["operation"] = "evaluate"
    elif mutation == "provider":
        state["provider"] = "aws-p5.48xlarge"
    elif mutation == "status":
        state["status"] = "Pending"
    elif mutation == "objects-collected":
        state["objects_collected"] = 15
    elif mutation == "bytes-collected-zero":
        state["bytes_collected"] = 0
    elif mutation == "bytes-collected-bool":
        state["bytes_collected"] = True
    elif mutation == "lifecycle-binding":
        state["purchase_model"] = ""
    elif mutation == "seed-key-mismatch":
        state["seed"] = 3
    elif mutation == "run-receipt-shape":
        state["run_receipt"].pop("bytes")
    elif mutation == "run-receipt-key":
        state["run_receipt"]["uri"] = (
            f"{S3_ROOT}/receipts/runs/seed-2/sha256/" + "f" * 64 + ".json"
        )
    elif mutation == "collection-receipt-shape":
        state["collection_receipt"]["version_id"] = "null"
    elif mutation == "collection-receipt-key":
        state["collection_receipt"]["uri"] = (
            f"{S3_ROOT}/receipts/runs/seed-2/sha256/"
            + state["collection_receipt"]["sha256"]
            + ".json"
        )
    elif mutation == "manifest-binding":
        written_manifest = hashlib.sha256(b"other-manifest").hexdigest()
    store = StateStore(tmp_path / "state")

    with store.locked(), pytest.raises(MsctlError) as caught:
        store.write_collection(written_manifest, state)

    assert caught.value.code in {"STATE_CORRUPT", "SCHEMA_INVALID"}, mutation
    with store.locked():
        assert store.read_collection(written_manifest) is None
