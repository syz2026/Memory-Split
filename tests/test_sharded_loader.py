import json
import os
import shutil

import numpy as np
import pytest
import torch

import corpusgen.parallel as parallel
from corpusgen.parallel import (
    FixtureRenderer,
    ParallelBuildConfig,
    build_parallel_corpus,
    fixture_catalog,
    render_metadata,
)
from corpusgen.parallel.canonical import canonical_json_bytes
from train.data import (
    PARALLEL_SIDECAR_V2_CONTRACT,
    BatchSlice,
    PackedShards,
)
from train.trainer import Trainer


def build_aligned_publication(tmp_path, *, record_count=14, update_tokens=32):
    root = tmp_path / f"parallel-{record_count}"
    build_parallel_corpus(
        fixture_catalog(record_count=record_count),
        FixtureRenderer(),
        ParallelBuildConfig(
            lane_weights=(("natural", 1), ("facts", 1), ("reasoning", 1)),
            update_tokens=update_tokens,
            allow_fewer_shards=True,
        ),
        root,
    )
    return root


def build_weighted_publication(tmp_path, *, record_count=13, update_tokens=32):
    root = tmp_path / f"weighted-parallel-{record_count}"
    catalog = fixture_catalog(record_count=record_count)
    renderer = FixtureRenderer()
    logical_tokens = sum(
        record.token_length for record in render_metadata(catalog, renderer)
    )
    dense = np.ones(logical_tokens, dtype=np.uint8)
    split90 = np.ones(logical_tokens, dtype=np.uint8)
    split90[::3] = 0
    values = {
        "dense_target_weights": dense,
        "split90_target_weights": split90,
    }
    paths = {}
    for name, weights in values.items():
        path = tmp_path / f"{name}-{record_count}.bin"
        weights.tofile(path)
        paths[name] = path
    receipt = build_parallel_corpus(
        catalog,
        renderer,
        ParallelBuildConfig(
            lane_weights=(("natural", 1), ("facts", 1), ("reasoning", 1)),
            update_tokens=update_tokens,
            allow_fewer_shards=True,
        ),
        root,
        sidecar_paths=paths,
    )
    padded = {
        name: np.concatenate(
            (
                weights,
                np.zeros(receipt["padding_tokens"], dtype=np.uint8),
            )
        )
        for name, weights in values.items()
    }
    return root, receipt, padded


def concatenate_shards(publication, destination):
    shard_paths = sorted((publication / "shards").glob("*.bin"))
    destination.write_bytes(b"".join(path.read_bytes() for path in shard_paths))
    return destination


def test_parallel_publication_matches_single_file_across_shard_boundary(tmp_path):
    publication = build_aligned_publication(tmp_path)
    legacy_path = concatenate_shards(publication, tmp_path / "legacy.bin")
    legacy = PackedShards(legacy_path, None, ctx=4, batch_size=2)
    sharded = PackedShards.from_parallel_corpus(
        publication,
        ctx=4,
        batch_size=2,
    )
    crossing = BatchSlice(global_start=28, sequence_count=2, ctx=4)

    legacy_batch = legacy.batch_from_slice(crossing)
    sharded_batch = sharded.batch_from_slice(crossing)

    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(legacy_batch, sharded_batch, strict=True)
    )
    assert sharded.provenance["kind"] == "parallel-publication"


def test_parallel_publication_verifies_shard_hashes_before_loading(tmp_path):
    publication = build_aligned_publication(tmp_path)
    shard = next((publication / "shards").glob("*.bin"))
    payload = bytearray(shard.read_bytes())
    payload[0] ^= 1
    shard.write_bytes(payload)

    with pytest.raises(ValueError, match="digest drift"):
        PackedShards.from_parallel_corpus(publication, ctx=4, batch_size=2)


def test_parallel_publication_rejects_noncanonical_receipt(tmp_path):
    publication = build_aligned_publication(tmp_path)
    receipt = publication / "receipt.json"
    receipt.write_bytes(receipt.read_bytes() + b" ")

    with pytest.raises(ValueError, match="receipt"):
        PackedShards.from_parallel_corpus(publication, ctx=4, batch_size=2)


def test_parallel_loader_opens_exact_supplied_receipt_file_and_relative_artifacts(
    tmp_path,
):
    publication, receipt, _ = build_weighted_publication(tmp_path)
    supplied_receipt = publication / "corpus-receipt.json"
    (publication / "receipt.json").rename(supplied_receipt)

    loader = PackedShards.from_parallel_corpus(
        supplied_receipt,
        ctx=4,
        batch_size=2,
        sidecar_name="split90_target_weights",
    )

    assert loader.provenance["build_id"] == receipt["build_id"]
    assert loader.target_weights is not None
    loader.close()


def test_parallel_loader_requires_strict_integer_receipt_scalars_even_if_preverified(
    tmp_path,
    monkeypatch,
):
    publication = build_aligned_publication(tmp_path)
    receipt_path = publication / "receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["logical_tokens"] = float(receipt["logical_tokens"])
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    monkeypatch.setattr(parallel, "verify_parallel_corpus", lambda _source: receipt)

    with pytest.raises(ValueError, match="logical_tokens.*integer"):
        PackedShards.from_parallel_corpus(
            receipt_path,
            ctx=4,
            batch_size=2,
        )


@pytest.mark.parametrize("relative", ["catalog.jsonl", "shards/shard-00000-of-00011.bin"])
def test_parallel_loader_rehashes_pinned_artifacts_after_semantic_verification(
    tmp_path,
    monkeypatch,
    relative,
):
    publication = build_aligned_publication(tmp_path)
    original_verify = parallel.verify_parallel_corpus

    def verify_then_replace(root, **kwargs):
        receipt = original_verify(root, **kwargs)
        artifact = publication / relative
        replacement = tmp_path / f"replacement-{artifact.name}"
        payload = bytearray(artifact.read_bytes())
        payload[0] ^= 1
        replacement.write_bytes(payload)
        os.replace(replacement, artifact)
        return receipt

    monkeypatch.setattr(parallel, "verify_parallel_corpus", verify_then_replace)

    with pytest.raises(ValueError, match="digest drift"):
        PackedShards.from_parallel_corpus(publication, ctx=4, batch_size=2)


def test_parallel_loader_rejects_receipt_symlink_swap_after_verification(
    tmp_path,
    monkeypatch,
):
    publication = build_aligned_publication(tmp_path)
    original_verify = parallel.verify_parallel_corpus

    def verify_then_swap_receipt(root, **kwargs):
        verified = original_verify(root, **kwargs)
        receipt = publication / "receipt.json"
        pinned_target = tmp_path / "moved-receipt.json"
        os.replace(receipt, pinned_target)
        receipt.symlink_to(pinned_target)
        return verified

    monkeypatch.setattr(parallel, "verify_parallel_corpus", verify_then_swap_receipt)

    with pytest.raises(ValueError, match="symlink|unsafe"):
        PackedShards.from_parallel_corpus(publication, ctx=4, batch_size=2)


def test_v2_loader_selects_ordered_descriptor_pinned_split90_sidecar(tmp_path):
    publication, receipt, sidecars = build_weighted_publication(tmp_path)
    loader = PackedShards.from_parallel_corpus(
        publication,
        ctx=4,
        batch_size=2,
        sidecar_name="split90_target_weights",
    )
    crossing = BatchSlice(global_start=28, sequence_count=2, ctx=4)

    _, targets, weights = loader.weighted_batch_from_slice(crossing)

    expected = torch.from_numpy(
        sidecars["split90_target_weights"][29:37]
        .reshape(2, 4)
        .astype(np.float32)
    )
    assert receipt["format"] == "memorysplit-parallel-corpus-v2"
    assert torch.equal(weights, expected)
    assert targets.numel() == 8
    assert len(loader.weight_paths) == receipt["shard_count"]
    assert loader.provenance["sidecar_name"] == "split90_target_weights"


@pytest.mark.parametrize("sidecar_name", [None, "unknown_target_weights"])
def test_v2_loader_requires_one_receipt_bound_sidecar_name(tmp_path, sidecar_name):
    publication, _, _ = build_weighted_publication(tmp_path)

    with pytest.raises(ValueError, match="sidecar_name"):
        PackedShards.from_parallel_corpus(
            publication,
            ctx=4,
            batch_size=2,
            sidecar_name=sidecar_name,
        )


def test_v2_loader_rehashes_pinned_sidecar_after_semantic_verification(
    tmp_path,
    monkeypatch,
):
    publication, receipt, _ = build_weighted_publication(tmp_path)
    split90 = next(
        record
        for record in receipt["sidecar_sets"]
        if record["name"] == "split90_target_weights"
    )
    relative = split90["artifacts"][0]["path"]
    original_verify = parallel.verify_parallel_corpus

    def verify_then_replace(root, **kwargs):
        verified = original_verify(root, **kwargs)
        artifact = publication / relative
        replacement = tmp_path / "replacement-sidecar.bin"
        payload = bytearray(artifact.read_bytes())
        payload[0] ^= 1
        replacement.write_bytes(payload)
        os.replace(replacement, artifact)
        return verified

    monkeypatch.setattr(parallel, "verify_parallel_corpus", verify_then_replace)

    with pytest.raises(ValueError, match="digest drift"):
        PackedShards.from_parallel_corpus(
            publication,
            ctx=4,
            batch_size=2,
            sidecar_name="split90_target_weights",
        )


@pytest.mark.parametrize("sidecar_name", ["mask_path", "weights_path"])
def test_parallel_v1_rejects_unbound_sidecars(tmp_path, sidecar_name):
    publication = build_aligned_publication(tmp_path)
    sidecar = tmp_path / f"{sidecar_name}.bin"
    np.ones(352, dtype=np.uint8).tofile(sidecar)

    with pytest.raises(ValueError, match="does not bind sidecars"):
        PackedShards.from_parallel_corpus(
            publication,
            ctx=4,
            batch_size=2,
            **{sidecar_name: sidecar},
        )


def test_future_parallel_sidecar_contract_is_receipt_bound_and_padding_safe():
    assert PARALLEL_SIDECAR_V2_CONTRACT == {
        "format": "memorysplit-parallel-corpus-v2",
        "receipt_field": "sidecar_sets",
        "artifact_fields": ("bytes", "path", "sha256"),
        "set_fields": (
            "artifacts",
            "dtype",
            "items",
            "name",
            "stream_sha256",
        ),
        "defined_names": (
            "dense_target_weights",
            "split90_target_weights",
        ),
        "alignment": {
            "items": "packed_tokens",
            "shard_order": "token_assignments",
        },
        "padding_rule": {
            "sidecar": "target_weights",
            "required_value": 0,
        },
    }


@pytest.mark.parametrize("sidecar_kind", ["mask", "weights"])
def test_legacy_sidecars_require_exact_token_length(tmp_path, sidecar_kind):
    tokens = tmp_path / "tokens.bin"
    sidecar = tmp_path / f"{sidecar_kind}.bin"
    np.arange(64, dtype=np.uint16).tofile(tokens)
    np.ones(63, dtype=np.uint8).tofile(sidecar)
    kwargs = (
        {"mask_path": sidecar}
        if sidecar_kind == "mask"
        else {"mask_path": None, "weights_path": sidecar}
    )

    with pytest.raises(ValueError, match=f"{sidecar_kind}.*length"):
        PackedShards(tokens, ctx=4, batch_size=2, **kwargs)


def test_legacy_provenance_is_content_bound_and_path_independent(tmp_path):
    tokens = tmp_path / "tokens.bin"
    copied = tmp_path / "copied.bin"
    np.arange(64, dtype=np.uint16).tofile(tokens)
    shutil.copyfile(tokens, copied)

    original = PackedShards(tokens, None, ctx=4, batch_size=2)
    relocated = PackedShards(copied, None, ctx=4, batch_size=2)

    assert original.provenance == relocated.provenance
    assert len(original.provenance["tokens"]["sha256"]) == 64
    assert "path" not in original.provenance["tokens"]


def test_parallel_publication_rejects_padding_without_bound_weights(tmp_path):
    publication = build_aligned_publication(
        tmp_path,
        record_count=13,
        update_tokens=32,
    )

    with pytest.raises(ValueError, match="padding"):
        PackedShards.from_parallel_corpus(publication, ctx=4, batch_size=2)


def test_training_update_must_match_publication_update_alignment(tmp_path):
    publication = build_aligned_publication(tmp_path)
    sharded = PackedShards.from_parallel_corpus(
        publication,
        ctx=4,
        batch_size=2,
    )

    with pytest.raises(ValueError, match="publication update_tokens"):
        sharded.validate_update_alignment(16)


def test_trainer_loads_verified_parallel_publication_without_concatenation(tmp_path):
    publication = build_aligned_publication(tmp_path)
    cfg = {
        "model": {
            "n_layer": 1,
            "n_head": 1,
            "d_model": 8,
            "ctx": 4,
            "vocab_size": 256,
        },
        "train_corpus": str(publication),
        "micro_batch_size": 2,
        "tokens_per_step": 32,
        "max_steps": 1,
        "lr": 1e-3,
        "seed": 9,
        "out_dir": str(tmp_path / "out"),
        "device": "cpu",
    }

    trainer = Trainer(cfg)

    assert trainer.data.provenance["kind"] == "parallel-publication"
    assert len(trainer.data.token_paths) > 1


def test_trainer_selects_verified_v2_target_weights_by_sidecar_name(tmp_path):
    publication, _, _ = build_weighted_publication(tmp_path)
    cfg = {
        "model": {
            "n_layer": 1,
            "n_head": 1,
            "d_model": 8,
            "ctx": 4,
            "vocab_size": 256,
        },
        "train_corpus": str(publication),
        "sidecar_name": "split90_target_weights",
        "micro_batch_size": 2,
        "tokens_per_step": 32,
        "max_steps": 1,
        "lr": 1e-3,
        "seed": 9,
        "out_dir": str(tmp_path / "weighted-out"),
        "device": "cpu",
    }

    trainer = Trainer(cfg)

    assert trainer.data.target_weights is not None
    assert trainer.data.provenance["sidecar_name"] == "split90_target_weights"
    trainer.close()


def _tiny_trainer_config(tmp_path, *, out_name):
    return {
        "model": {
            "n_layer": 1,
            "n_head": 1,
            "d_model": 8,
            "ctx": 4,
            "vocab_size": 256,
        },
        "micro_batch_size": 2,
        "tokens_per_step": 32,
        "max_steps": 1,
        "lr": 1e-3,
        "seed": 9,
        "out_dir": str(tmp_path / out_name),
        "device": "cpu",
    }


def test_trainer_rejects_sidecar_name_with_legacy_train_bin(tmp_path):
    tokens = tmp_path / "legacy.bin"
    np.arange(64, dtype=np.uint16).tofile(tokens)
    cfg = _tiny_trainer_config(tmp_path, out_name="legacy-sidecar-name")
    cfg.update(
        {
            "train_bin": str(tokens),
            "sidecar_name": "dense_target_weights",
        }
    )

    with pytest.raises(ValueError, match="sidecar_name.*train_bin"):
        Trainer(cfg)

    assert not (tmp_path / "legacy-sidecar-name").exists()


def test_trainer_rejects_split90_label_without_receipt_v2_sidecar(tmp_path):
    publication = build_aligned_publication(tmp_path)
    cfg = _tiny_trainer_config(tmp_path, out_name="unbound-split90")
    cfg.update(
        {
            "condition": "split90",
            "train_corpus": str(publication / "receipt.json"),
        }
    )

    with pytest.raises(ValueError, match="Split90.*receipt-v2 sidecar"):
        Trainer(cfg)

    assert not (tmp_path / "unbound-split90").exists()


def test_trainer_rejects_split90_label_with_dense_receipt_sidecar(tmp_path):
    publication, _, _ = build_weighted_publication(tmp_path)
    cfg = _tiny_trainer_config(tmp_path, out_name="dense-labeled-split90")
    cfg.update(
        {
            "condition": "split90",
            "train_corpus": str(publication / "receipt.json"),
            "sidecar_name": "dense_target_weights",
        }
    )

    with pytest.raises(ValueError, match="Split90.*split90_target_weights"):
        Trainer(cfg)

    assert not (tmp_path / "dense-labeled-split90").exists()


def test_trainer_rejects_ambiguous_legacy_and_parallel_sources(tmp_path):
    publication = build_aligned_publication(tmp_path)
    legacy = concatenate_shards(publication, tmp_path / "legacy.bin")
    cfg = {
        "model": {
            "n_layer": 1,
            "n_head": 1,
            "d_model": 8,
            "ctx": 4,
            "vocab_size": 256,
        },
        "train_bin": str(legacy),
        "train_corpus": str(publication),
        "micro_batch_size": 2,
        "tokens_per_step": 32,
        "max_steps": 1,
        "lr": 1e-3,
        "seed": 9,
        "out_dir": str(tmp_path / "out"),
        "device": "cpu",
    }

    with pytest.raises(ValueError, match="exactly one"):
        Trainer(cfg)
