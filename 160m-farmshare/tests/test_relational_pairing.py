from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json

import pytest

from evals.checkpoint_binding import canonical_configuration_sha256
from tests.task8_helpers import make_rows, make_summary, replace_row


EXPECTED_PAIR_FIELDS = (
    "split_checkpoint_sha256",
    "dense_checkpoint_sha256",
    "model_id",
    "seed",
    "raw_token_count",
    "evaluator_sha256",
    "data_sha256",
    "relation_schema_sha256",
    "split_configuration_sha256",
    "dense_configuration_sha256",
    "result_schema_sha256",
    "split_result_provenance_sha256",
    "dense_result_provenance_sha256",
    "study_provenance_sha256",
    "split_pair_fingerprint",
    "dense_pair_fingerprint",
)


def _pairing_module():
    return importlib.import_module("evals.relational_pairing")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _configs():
    common = {
        "model": "d160m",
        "seed": 1001,
        "load": "n800k",
        "ctx": 128,
        "tokens_per_step": 10,
        "initialization_seed": 1001,
        "data_seed": 17,
        "train_bin": "train.bin",
        "train_mask": "train.mask",
        "packing": {"block_size": 128},
        "optimizer": {"name": "adamw"},
        "scheduler": {"name": "cosine"},
        "raw_positions": {"start": 0},
        "decode_budget": 6,
        "checkpoint_schedule": [5, 10, 20],
    }
    return {
        "split": {
            **common,
            "condition": "split",
            "train_weights": "split.weights.bin",
        },
        "dense": {
            **common,
            "condition": "dense",
            "train_weights": "dense.weights.bin",
        },
    }


def _paired_inputs():
    configs = _configs()
    shared = {
        "model_id": "d160m",
        "seed": 1001,
        "raw_token_count": 100,
        "evaluator_sha256": _sha("evaluator"),
        "data_sha256": _sha("data"),
        "relation_schema_sha256": _sha("relation-schema"),
        "result_schema_sha256": _sha("result-schema"),
    }
    anchors = {}
    for arm in ("split", "dense"):
        rows = make_rows(
            f"pairing-{arm}",
            arm,
            seeds=(1001,),
            pair_counts={
                "path_composition": 1,
                "date_ordering": 1,
                "balanced_equality": 1,
            },
        )
        identity = {
            **shared,
            "checkpoint_sha256": _sha(f"checkpoint-{arm}"),
            "configuration_sha256": canonical_configuration_sha256(
                configs[arm]
            ),
            "provenance_sha256": _sha(f"provenance-{arm}"),
        }
        anchors[arm] = make_summary(
            tuple(replace_row(row, **identity) for row in rows)
        )
    return anchors["split"], anchors["dense"], configs["split"], configs["dense"]


def _canonical_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _rehash(raw: dict) -> None:
    raw["receipt_sha256"] = hashlib.sha256(
        _canonical_bytes(
            {
                key: value
                for key, value in raw.items()
                if key != "receipt_sha256"
            }
        )
    ).hexdigest()


def test_split_evaluation_produces_pairing_receipt_without_test_helpers(tmp_path):
    pairing = _pairing_module()
    split, dense, split_config, dense_config = _paired_inputs()
    receipt = pairing.build_pairing_receipt(
        split,
        dense,
        split_config,
        dense_config,
    )
    eval_dir = tmp_path / "split-run" / "evals"
    eval_dir.mkdir(parents=True)

    path = pairing.publish_pairing_receipt(
        eval_dir / "pairing-receipt.json",
        receipt,
    )

    assert pairing.validate_pairing_receipt(
        json.loads(path.read_text())
    ) == receipt
    assert path.read_bytes() == _canonical_bytes(receipt.to_dict())
    assert pairing.PAIR_FIELDS == EXPECTED_PAIR_FIELDS
    assert not list(eval_dir.glob(".pairing-receipt.*"))


def test_pairing_receipt_binds_validated_anchor_and_config_fields():
    pairing = _pairing_module()
    split, dense, split_config, dense_config = _paired_inputs()

    receipt = pairing.build_pairing_receipt(
        split,
        dense,
        split_config,
        dense_config,
    )
    raw = receipt.to_dict()

    assert {field: raw[field] for field in EXPECTED_PAIR_FIELDS[:-3]} == {
        "split_checkpoint_sha256": split.checkpoint_sha256,
        "dense_checkpoint_sha256": dense.checkpoint_sha256,
        "model_id": split.model_id,
        "seed": split.seed,
        "raw_token_count": split.raw_token_count,
        "evaluator_sha256": split.evaluator_sha256,
        "data_sha256": split.data_sha256,
        "relation_schema_sha256": split.relation_schema_sha256,
        "split_configuration_sha256": split.configuration_sha256,
        "dense_configuration_sha256": dense.configuration_sha256,
        "result_schema_sha256": split.result_schema_sha256,
        "split_result_provenance_sha256": split.provenance_sha256,
        "dense_result_provenance_sha256": dense.provenance_sha256,
    }
    assert raw["split_pair_fingerprint"] != raw["dense_pair_fingerprint"]

    with pytest.raises(ValueError, match="data_sha256"):
        pairing.build_pairing_receipt(
            split,
            replace(dense, data_sha256=_sha("crossed-data")),
            split_config,
            dense_config,
        )
    with pytest.raises(ValueError, match="configuration"):
        pairing.build_pairing_receipt(
            split,
            dense,
            split_config,
            {**dense_config, "load": "n50k"},
        )


@pytest.mark.parametrize("field", EXPECTED_PAIR_FIELDS)
def test_pairing_receipt_rejects_every_pair_fingerprint_mismatch(field):
    pairing = _pairing_module()
    split, dense, split_config, dense_config = _paired_inputs()
    receipt = pairing.build_pairing_receipt(
        split,
        dense,
        split_config,
        dense_config,
    )
    raw = receipt.to_dict()
    if field in {"seed", "raw_token_count"}:
        raw[field] += 1
    elif field == "model_id":
        raw[field] = "crossed-model"
    else:
        raw[field] = (
            "f" * 64 if raw[field] != "f" * 64 else "e" * 64
        )
    _rehash(raw)

    with pytest.raises(ValueError, match="mismatch|fingerprint"):
        pairing.validate_pairing_receipt(raw)


def test_pairing_receipt_publication_rejects_overwrite_traversal_and_symlinks(
    tmp_path,
):
    pairing = _pairing_module()
    split, dense, split_config, dense_config = _paired_inputs()
    receipt = pairing.build_pairing_receipt(
        split,
        dense,
        split_config,
        dense_config,
    )
    eval_dir = tmp_path / "split-run" / "evals"
    eval_dir.mkdir(parents=True)
    output = eval_dir / "pairing-receipt.json"

    pairing.publish_pairing_receipt(output, receipt)
    with pytest.raises(FileExistsError, match="already exists"):
        pairing.publish_pairing_receipt(output, receipt)

    with pytest.raises(ValueError, match="canonical|traversal"):
        pairing.publish_pairing_receipt(
            eval_dir / ".." / "pairing-receipt.json",
            receipt,
        )

    output.unlink()
    external = tmp_path / "external.json"
    external.write_text("attacker")
    output.symlink_to(external)
    with pytest.raises(FileExistsError, match="already exists"):
        pairing.publish_pairing_receipt(output, receipt)
    assert external.read_text() == "attacker"

    aliased_parent = tmp_path / "aliased-evals"
    aliased_parent.symlink_to(eval_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical|symlink"):
        pairing.publish_pairing_receipt(
            aliased_parent / "pairing-receipt.json",
            receipt,
        )
