from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import cluster.corpus_contract as bridge
from corpusgen.parallel import (
    CatalogRecord,
    InputCatalog,
    ParallelBuildConfig,
    RenderedRecord,
    build_parallel_corpus,
)
from corpusgen.parallel.canonical import canonical_json_bytes
from train.data import PackedShards


ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = ROOT / "configs" / "reasoning-dataset-v2.json"
TINY_TOTAL = 48
TINY_UPDATE = 8
TINY_STEPS = 6
TINY_QUOTAS = {
    "fineweb_edu": 12,
    "finemath": 7,
    "wikidata_graph": 10,
    "synthetic_graph": 5,
    "verified_synthetic_multihop": 7,
    "wikidata_path_reasoning": 4,
    "relational_refinement": 1,
    "objective_auxiliary": 2,
}
LANE_WEIGHTS = tuple(TINY_QUOTAS.items())
DYNAMIC_POINTER_FIELDS = (
    "expected_receipt_sha256",
    "expected_source_receipt_sha256",
    "expected_ordered_token_stream_sha256",
    "expected_packed_stream_sha256",
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class QuotaRenderer:
    renderer_id = "135m-bridge-test-u16-v1"

    def render(self, record: CatalogRecord) -> RenderedRecord:
        count = int(record.payload.decode("ascii"))
        return RenderedRecord(
            token_ids=(record.ordinal + 1,) * count,
            flags=("test-verified",),
        )


def _write_tiny_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str]:
    raw = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    recipe = raw["sprint_recipe"]
    recipe["targets_per_update"] = TINY_UPDATE
    recipe["optimizer_steps"] = TINY_STEPS
    recipe["raw_target_tokens"] = TINY_TOTAL
    allocation = recipe["realized_token_allocation"]
    allocation["total_tokens"] = TINY_TOTAL
    allocation["token_quotas"] = dict(TINY_QUOTAS)
    data = (json.dumps(raw, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = tmp_path / "configs" / "reasoning-dataset-v2.json"
    path.parent.mkdir()
    path.write_bytes(data)
    digest = _sha(data)
    monkeypatch.setattr(bridge, "RAW_TARGETS", TINY_TOTAL)
    monkeypatch.setattr(bridge, "TARGETS_PER_UPDATE", TINY_UPDATE)
    monkeypatch.setattr(bridge, "TERMINAL_UPDATES", TINY_STEPS)
    monkeypatch.setattr(bridge, "AUTHORITATIVE_RECIPE_SHA256", digest)
    return path, digest


def _build_task4_publication(
    root: Path,
    *,
    quotas: dict[str, int] | None = None,
) -> dict:
    realized = dict(quotas or TINY_QUOTAS)
    records = tuple(
        CatalogRecord(
            ordinal=index,
            record_id=f"record-{index:02d}",
            lane=lane,
            source=f"source-{lane}",
            source_key=f"row-{index}",
            payload=str(realized[lane]).encode("ascii"),
        )
        for index, lane in enumerate(TINY_QUOTAS)
    )
    catalog = InputCatalog(records)
    support = root.with_name(f"{root.name}-inputs")
    support.mkdir()
    dense = support / "dense.bin"
    split = support / "split90.bin"
    dense.write_bytes(bytes([1]) * sum(realized.values()))
    split.write_bytes(
        bytes(0 if index % 4 == 0 else 1 for index in range(sum(realized.values())))
    )
    receipt = build_parallel_corpus(
        catalog,
        QuotaRenderer(),
        ParallelBuildConfig(
            lane_weights=LANE_WEIGHTS,
            update_tokens=TINY_UPDATE,
            shard_count=2,
        ),
        root,
        sidecar_paths={
            "dense_target_weights": dense,
            "split90_target_weights": split,
        },
    )
    return dict(receipt)


def _write_pointer_template(path: Path, recipe_sha256: str) -> Path:
    raw = json.loads(
        (ROOT / "DATASET-POINTER-SLURM-135M.json").read_text(encoding="utf-8")
    )
    raw["expected_source_lock_sha256"] = recipe_sha256
    path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def tiny_task4(tmp_path, monkeypatch):
    recipe, recipe_sha = _write_tiny_recipe(tmp_path, monkeypatch)
    publication = tmp_path / "task4-publication"
    receipt = _build_task4_publication(publication)
    return {
        "publication": publication,
        "receipt": receipt,
        "receipt_sha": _sha((publication / "receipt.json").read_bytes()),
        "ordered_sha": receipt["ordered_stream_sha256"],
        "recipe": recipe,
        "recipe_sha": recipe_sha,
    }


def test_authoritative_recipe_is_exact_upstream_and_has_no_invented_lane_hashes():
    data = SOURCE_LOCK.read_bytes()
    raw = json.loads(data)
    lanes = raw["sprint_recipe"]["lanes"]

    assert _sha(data) == bridge.AUTHORITATIVE_RECIPE_SHA256
    assert [lane["id"] for lane in lanes] == list(TINY_QUOTAS)
    assert [lane["share_percent"] for lane in lanes] == [
        25.0,
        15.0,
        20.0,
        10.0,
        15.0,
        7.5,
        2.5,
        5.0,
    ]
    assert all("source_lock_sha256" not in lane for lane in lanes)


def test_checked_in_pointer_is_unfrozen_and_contains_no_speculative_hashes():
    path = ROOT / "DATASET-POINTER-SLURM-135M.json"
    raw = json.loads(path.read_text(encoding="utf-8"))

    pointer = bridge.load_dataset_pointer(path, require_frozen=False)
    assert raw["schema_version"] == 2
    assert raw["layout_format"] == bridge.BRIDGE_FORMAT
    assert raw["expected_source_lock_sha256"] == bridge.AUTHORITATIVE_RECIPE_SHA256
    assert all(raw[field] is None for field in DYNAMIC_POINTER_FIELDS)
    assert pointer.expected_receipt_sha256 == ""
    with pytest.raises(bridge.CorpusContractError, match="unfrozen"):
        bridge.load_dataset_pointer(path)


def test_task4_publication_materializes_freezes_verifies_and_stages(tiny_task4, tmp_path):
    source = bridge.verify_task4_publication(
        tiny_task4["publication"],
        source_lock_path=tiny_task4["recipe"],
        expected_source_lock_sha256=tiny_task4["recipe_sha"],
        expected_receipt_sha256=tiny_task4["receipt_sha"],
        expected_ordered_sha256=tiny_task4["ordered_sha"],
    )
    dataset = tmp_path / "dataset"
    converted = bridge.materialize_135m_layout(
        tiny_task4["publication"],
        dataset,
        source_lock_path=tiny_task4["recipe"],
        expected_source_lock_sha256=tiny_task4["recipe_sha"],
        expected_source_receipt_sha256=tiny_task4["receipt_sha"],
        expected_ordered_sha256=tiny_task4["ordered_sha"],
    )

    shard_records = [
        item
        for item in tiny_task4["receipt"]["artifacts"]
        if item["path"].startswith("shards/")
    ]
    expected_packed = b"".join(
        (tiny_task4["publication"] / item["path"]).read_bytes()
        for item in shard_records
    )
    assert (dataset / "packed" / "targets.bin").read_bytes() == expected_packed
    assert converted.source_receipt_sha256 == source.receipt_sha256
    assert converted.packed_stream_sha256 == source.packed_stream_sha256
    assert converted.raw_target_tokens == TINY_TOTAL
    loader = PackedShards(
        dataset / "packed" / "targets.bin",
        dataset / "sidecars" / "dense_target_weights.bin",
        ctx=4,
        batch_size=2,
    )
    observed_targets = []
    for _ in range(TINY_STEPS):
        _x, y = loader.next_batch()
        observed_targets.extend(y.flatten().tolist())
    packed_tokens = np.frombuffer(expected_packed, dtype=np.uint16).tolist()
    assert observed_targets == [*packed_tokens[1:], packed_tokens[0]]
    assert loader.state_dict() == {"cursor": TINY_TOTAL, "epoch": 1}

    template = _write_pointer_template(
        tmp_path / "pointer-template.json",
        tiny_task4["recipe_sha"],
    )
    frozen = bridge.freeze_dataset_pointer(
        tiny_task4["publication"],
        dataset,
        template_path=template,
        output_path=tmp_path / "pointer-frozen.json",
        expected_source_receipt_sha256=tiny_task4["receipt_sha"],
        expected_ordered_sha256=tiny_task4["ordered_sha"],
        source_lock_path=tiny_task4["recipe"],
    )
    pointer = bridge.load_dataset_pointer(frozen)
    assert pointer.expected_source_receipt_sha256 == tiny_task4["receipt_sha"]
    assert pointer.expected_ordered_token_stream_sha256 == tiny_task4["ordered_sha"]
    assert pointer.expected_packed_stream_sha256 == source.packed_stream_sha256
    assert pointer.expected_receipt_sha256 == converted.receipt_sha256

    verified = bridge.verify_dataset_root(
        dataset,
        pointer_path=frozen,
        source_lock_path=tiny_task4["recipe"],
    )
    assert verified.identity() == converted.identity()

    mirror = tmp_path / "site-mirror"
    staged = bridge.stage_dataset_no_replace(
        dataset,
        mirror,
        pointer_path=frozen,
        source_lock_path=tiny_task4["recipe"],
    )
    assert staged.identity() == converted.identity()
    assert Path(staged.receipt_path).parent == mirror
    with pytest.raises(FileExistsError):
        bridge.stage_dataset_no_replace(
            dataset,
            mirror,
            pointer_path=frozen,
            source_lock_path=tiny_task4["recipe"],
        )


def test_source_receipt_and_ordered_hashes_must_match_actual_bytes(tiny_task4):
    common = {
        "publication_root": tiny_task4["publication"],
        "source_lock_path": tiny_task4["recipe"],
        "expected_source_lock_sha256": tiny_task4["recipe_sha"],
    }
    with pytest.raises(bridge.CorpusContractError, match="authoritative Task-4"):
        bridge.verify_task4_publication(
            **common,
            expected_receipt_sha256="0" * 64,
            expected_ordered_sha256=tiny_task4["ordered_sha"],
        )
    with pytest.raises(bridge.CorpusContractError, match="authoritative Task-4"):
        bridge.verify_task4_publication(
            **common,
            expected_receipt_sha256=tiny_task4["receipt_sha"],
            expected_ordered_sha256="0" * 64,
        )


def test_authoritative_lane_quota_drift_is_rejected(tmp_path, monkeypatch):
    recipe, recipe_sha = _write_tiny_recipe(tmp_path, monkeypatch)
    publication = tmp_path / "wrong-quotas"
    wrong = dict(TINY_QUOTAS)
    wrong["fineweb_edu"] -= 1
    wrong["finemath"] += 1
    receipt = _build_task4_publication(publication, quotas=wrong)

    with pytest.raises(bridge.CorpusContractError, match="lane quotas"):
        bridge.verify_task4_publication(
            publication,
            source_lock_path=recipe,
            expected_source_lock_sha256=recipe_sha,
            expected_receipt_sha256=_sha((publication / "receipt.json").read_bytes()),
            expected_ordered_sha256=receipt["ordered_stream_sha256"],
        )


def test_freezer_rejects_flat_receipt_with_forged_source_sidecar_binding(
    tiny_task4,
    tmp_path,
):
    dataset = tmp_path / "dataset"
    bridge.materialize_135m_layout(
        tiny_task4["publication"],
        dataset,
        source_lock_path=tiny_task4["recipe"],
        expected_source_lock_sha256=tiny_task4["recipe_sha"],
        expected_source_receipt_sha256=tiny_task4["receipt_sha"],
        expected_ordered_sha256=tiny_task4["ordered_sha"],
    )
    sidecar = dataset / "sidecars" / "split90_target_weights.bin"
    payload = bytearray(sidecar.read_bytes())
    payload[0] ^= 1
    sidecar.write_bytes(payload)
    forged_sha = _sha(payload)
    receipt_path = dataset / "receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["task4_publication"]["sidecar_stream_sha256"][
        "split90_target_weights"
    ] = forged_sha
    artifact = next(
        item
        for item in receipt["artifacts"]
        if item["path"] == "sidecars/split90_target_weights.bin"
    )
    artifact["sha256"] = forged_sha
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    template = _write_pointer_template(
        tmp_path / "template.json",
        tiny_task4["recipe_sha"],
    )
    output = tmp_path / "must-not-freeze.json"

    with pytest.raises(
        bridge.CorpusContractError,
        match="every Task-4 source identity",
    ):
        bridge.freeze_dataset_pointer(
            tiny_task4["publication"],
            dataset,
            template_path=template,
            output_path=output,
            expected_source_receipt_sha256=tiny_task4["receipt_sha"],
            expected_ordered_sha256=tiny_task4["ordered_sha"],
            source_lock_path=tiny_task4["recipe"],
        )
    assert not output.exists()


def test_materializer_and_freezer_are_no_replace(tiny_task4, tmp_path):
    dataset = tmp_path / "dataset"
    kwargs = {
        "source_lock_path": tiny_task4["recipe"],
        "expected_source_lock_sha256": tiny_task4["recipe_sha"],
        "expected_source_receipt_sha256": tiny_task4["receipt_sha"],
        "expected_ordered_sha256": tiny_task4["ordered_sha"],
    }
    bridge.materialize_135m_layout(
        tiny_task4["publication"],
        dataset,
        **kwargs,
    )
    with pytest.raises(FileExistsError):
        bridge.materialize_135m_layout(
            tiny_task4["publication"],
            dataset,
            **kwargs,
        )

    template = _write_pointer_template(
        tmp_path / "template.json",
        tiny_task4["recipe_sha"],
    )
    output = tmp_path / "existing-pointer.json"
    output.write_text("do not replace", encoding="utf-8")
    with pytest.raises(FileExistsError):
        bridge.freeze_dataset_pointer(
            tiny_task4["publication"],
            dataset,
            template_path=template,
            output_path=output,
            expected_source_receipt_sha256=tiny_task4["receipt_sha"],
            expected_ordered_sha256=tiny_task4["ordered_sha"],
            source_lock_path=tiny_task4["recipe"],
        )
    assert output.read_text(encoding="utf-8") == "do not replace"


def test_frozen_flat_mirror_rejects_tampering_and_symlinks(tiny_task4, tmp_path):
    dataset = tmp_path / "dataset"
    bridge.materialize_135m_layout(
        tiny_task4["publication"],
        dataset,
        source_lock_path=tiny_task4["recipe"],
        expected_source_lock_sha256=tiny_task4["recipe_sha"],
        expected_source_receipt_sha256=tiny_task4["receipt_sha"],
        expected_ordered_sha256=tiny_task4["ordered_sha"],
    )
    template = _write_pointer_template(
        tmp_path / "template.json",
        tiny_task4["recipe_sha"],
    )
    pointer = bridge.freeze_dataset_pointer(
        tiny_task4["publication"],
        dataset,
        template_path=template,
        output_path=tmp_path / "frozen.json",
        expected_source_receipt_sha256=tiny_task4["receipt_sha"],
        expected_ordered_sha256=tiny_task4["ordered_sha"],
        source_lock_path=tiny_task4["recipe"],
    )
    packed = dataset / "packed" / "targets.bin"
    original = packed.read_bytes()
    packed.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(bridge.CorpusContractError, match="stream differs"):
        bridge.verify_dataset_root(
            dataset,
            pointer_path=pointer,
            source_lock_path=tiny_task4["recipe"],
        )

    packed.unlink()
    victim = tmp_path / "victim.bin"
    victim.write_bytes(original)
    packed.symlink_to(victim)
    with pytest.raises(bridge.CorpusContractError, match="symlink"):
        bridge.verify_dataset_root(
            dataset,
            pointer_path=pointer,
            source_lock_path=tiny_task4["recipe"],
        )


def test_unfrozen_pointer_rejects_speculative_dynamic_hash(tmp_path):
    raw = json.loads(
        (ROOT / "DATASET-POINTER-SLURM-135M.json").read_text(encoding="utf-8")
    )
    raw["expected_receipt_sha256"] = "0" * 64
    path = tmp_path / "speculative.json"
    path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(bridge.CorpusContractError, match="speculative"):
        bridge.load_dataset_pointer(path, require_frozen=False)
