from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from evals.confirmatory import sealing
from evals.confirmatory.contracts import (
    canonical_json_bytes,
    store_content_sha256,
)
from evals.confirmatory.fixtures import positive_fixture
from evals.confirmatory.sealing import (
    SEALED_RELEASE_SCHEMA,
    SealingError,
    preflight_model_visible_release,
    seal_release,
    verify_release,
)
from evals.confirmatory.study_lock import (
    FROZEN_PREREGISTRATION_SHA256_V3,
    REQUIRED_CONTROL_IDS,
    REQUIRED_FAMILIES,
    REQUIRED_MEMORY_MODES,
    REQUIRED_STRATA,
)
from scripts import seal_confirmatory_release as sealing_cli


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPOSITORY_ROOT / "configs" / "preregistration-v3.yaml"


def _write_source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    source.chmod(0o700)
    fixture = positive_fixture()
    for name in ("items.jsonl", "stores.jsonl", "sealed-gold.jsonl"):
        source.joinpath(name).write_bytes(fixture.artifacts[name])
    preregistration = tmp_path / "preregistration-v3.yaml"
    preregistration.write_bytes(PREREGISTRATION.read_bytes())
    return source, preregistration


def _publish_fixture(tmp_path: Path):
    source, preregistration = _write_source(tmp_path)
    output_root = tmp_path / "releases"
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    published = seal_release(
        source_dir=source,
        preregistration_path=preregistration,
        output_root=output_root,
        apply=True,
    )
    return source, preregistration, output_root, published


def _rewrite_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(record) for record in records))


def _write_file_at(directory_fd: int, name: str, content: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        os.write(descriptor, content)
    finally:
        os.close(descriptor)


def test_dry_run_builds_one_deterministic_v3_manifest_without_writing(tmp_path):
    source, preregistration = _write_source(tmp_path)
    output_root = tmp_path / "releases"

    first = seal_release(
        source_dir=source,
        preregistration_path=preregistration,
        output_root=output_root,
        apply=False,
    )
    second = seal_release(
        source_dir=source,
        preregistration_path=preregistration,
        output_root=tmp_path / "another-output",
        apply=False,
    )

    assert not output_root.exists()
    assert not first.published
    assert first.manifest_bytes == second.manifest_bytes
    assert first.release_sha256 == second.release_sha256
    assert first.release_sha256 == hashlib.sha256(first.manifest_bytes).hexdigest()
    assert first.release_id == (
        f"memorysplit-confirmatory-sealed-v3-{first.release_sha256}"
    )
    assert first.release_dir == output_root / first.release_id

    manifest = json.loads(first.manifest_bytes)
    assert canonical_json_bytes(manifest) == first.manifest_bytes
    assert manifest["record_type"] == SEALED_RELEASE_SCHEMA
    assert manifest["schema_version"] == 3
    assert manifest["preregistration"] == {
        "path": "configs/preregistration-v3.yaml",
        "sha256": FROZEN_PREREGISTRATION_SHA256_V3,
    }
    assert manifest["sealed_gold"] == {
        "path": "sealed-gold.jsonl",
        "sha256": hashlib.sha256(
            source.joinpath("sealed-gold.jsonl").read_bytes()
        ).hexdigest(),
    }
    visible = manifest["model_visible"]
    assert visible["items"]["sha256"] == hashlib.sha256(
        source.joinpath("items.jsonl").read_bytes()
    ).hexdigest()
    assert visible["stores"]["sha256"] == hashlib.sha256(
        source.joinpath("stores.jsonl").read_bytes()
    ).hexdigest()
    assert visible["item_ids"] == sorted(visible["item_ids"])
    assert visible["pair_ids"] == sorted(visible["pair_ids"])
    assert visible["world_ids"] == sorted(visible["world_ids"])
    assert visible["store_ids"] == sorted(visible["store_ids"])
    assert visible["item_count"] == len(visible["item_ids"])
    assert visible["pair_count"] == len(visible["pair_ids"])
    assert visible["world_count"] == len(visible["world_ids"])
    assert visible["store_count"] == len(visible["store_ids"])

    coverage = visible["coverage"]
    assert coverage["families"] == list(REQUIRED_FAMILIES)
    assert coverage["strata"] == list(REQUIRED_STRATA)
    assert coverage["memory_modes"] == list(REQUIRED_MEMORY_MODES)
    assert coverage["controls"] == list(REQUIRED_CONTROL_IDS)
    assert len(coverage["cells"]) == (
        len(REQUIRED_FAMILIES)
        * len(REQUIRED_STRATA)
        * len(REQUIRED_CONTROL_IDS)
    )
    assert all(cell["pair_count"] >= 1 for cell in coverage["cells"])
    assert all(cell["item_count"] == 2 * cell["pair_count"] for cell in coverage["cells"])


def test_sealing_rejects_incomplete_required_coverage(tmp_path):
    source, preregistration = _write_source(tmp_path)
    item_path = source / "items.jsonl"
    items = [json.loads(line) for line in item_path.read_bytes().splitlines()]
    removed_pair = items[-1]["pair_id"]
    _rewrite_jsonl(
        item_path,
        [item for item in items if item["pair_id"] != removed_pair],
    )

    gold_path = source / "sealed-gold.jsonl"
    gold = [json.loads(line) for line in gold_path.read_bytes().splitlines()]
    _rewrite_jsonl(
        gold_path,
        [record for record in gold if record["pair_id"] != removed_pair],
    )

    store_id = next(
        item["store_id"] for item in items if item["pair_id"] == removed_pair
    )
    store_path = source / "stores.jsonl"
    stores = [json.loads(line) for line in store_path.read_bytes().splitlines()]
    _rewrite_jsonl(
        store_path,
        [store for store in stores if store["store_id"] != store_id],
    )

    with pytest.raises(ValueError, match="coverage|required"):
        seal_release(
            source_dir=source,
            preregistration_path=preregistration,
            output_root=tmp_path / "releases",
            apply=False,
        )


def test_sealing_rejects_gold_that_the_registered_solver_cannot_reproduce(tmp_path):
    source, preregistration = _write_source(tmp_path)
    gold_path = source / "sealed-gold.jsonl"
    gold = [json.loads(line) for line in gold_path.read_bytes().splitlines()]
    gold[0]["answer"] = "forged"
    _rewrite_jsonl(gold_path, gold)

    with pytest.raises(ValueError, match="gold|solver"):
        seal_release(
            source_dir=source,
            preregistration_path=preregistration,
            output_root=tmp_path / "releases",
            apply=False,
        )


@pytest.mark.parametrize("mutation", ["insert", "remove", "replace"])
def test_seal_plan_detects_source_membership_mutation_after_solver_replay(
    tmp_path,
    monkeypatch,
    mutation,
):
    source, preregistration = _write_source(tmp_path)
    items_bytes = source.joinpath("items.jsonl").read_bytes()

    def mutate_source(event, **_context):
        if event != "seal_plan_before_final_return":
            return
        if mutation == "insert":
            source.joinpath("unexpected").write_text("inserted", encoding="utf-8")
        elif mutation == "remove":
            source.joinpath("stores.jsonl").rename(
                tmp_path / "removed-source-stores.jsonl"
            )
        else:
            items = source / "items.jsonl"
            items.rename(tmp_path / "displaced-source-items.jsonl")
            items.write_bytes(items_bytes)
            items.chmod(0o644)

    monkeypatch.setattr(sealing, "_run_mutation_hook", mutate_source)

    with pytest.raises(
        SealingError,
        match="changed|membership|replaced|entries",
    ):
        seal_release(
            source_dir=source,
            preregistration_path=preregistration,
            output_root=tmp_path / "releases",
            apply=False,
        )


def test_counterfactual_twins_may_use_distinct_stores_in_the_same_world(tmp_path):
    source, preregistration = _write_source(tmp_path)
    item_path = source / "items.jsonl"
    items = [json.loads(line) for line in item_path.read_bytes().splitlines()]
    counterfactual = next(
        item for item in items if item["twin"] == "counterfactual"
    )
    old_store_id = counterfactual["store_id"]
    new_store_id = f"{old_store_id}-counterfactual"
    counterfactual["store_id"] = new_store_id
    _rewrite_jsonl(item_path, items)

    store_path = source / "stores.jsonl"
    stores = [json.loads(line) for line in store_path.read_bytes().splitlines()]
    original_store = next(
        store for store in stores if store["store_id"] == old_store_id
    )
    changed_store = {
        **original_store,
        "store_id": new_store_id,
    }
    changed_store["content_sha256"] = store_content_sha256(
        new_store_id,
        changed_store["world_id"],
        changed_store["rows"],
    )
    stores.append(changed_store)
    stores.sort(key=lambda store: store["store_id"])
    _rewrite_jsonl(store_path, stores)

    gold_path = source / "sealed-gold.jsonl"
    gold = [json.loads(line) for line in gold_path.read_bytes().splitlines()]
    matching_gold = next(
        record for record in gold if record["item_id"] == counterfactual["item_id"]
    )
    matching_gold["store_sha256"] = changed_store["content_sha256"]
    _rewrite_jsonl(gold_path, gold)

    result = seal_release(
        source_dir=source,
        preregistration_path=preregistration,
        output_root=tmp_path / "releases",
        apply=False,
    )

    assert result.store_count == len(stores)


def test_apply_publishes_exact_v2_bytes_in_one_content_addressed_directory(
    tmp_path,
):
    source, preregistration = _write_source(tmp_path)
    output_root = tmp_path / "releases"
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)

    result = seal_release(
        source_dir=source,
        preregistration_path=preregistration,
        output_root=output_root,
        apply=True,
    )

    assert result.published
    assert result.release_dir == output_root / result.release_id
    assert {path.name for path in result.release_dir.iterdir()} == {
        "items.jsonl",
        "stores.jsonl",
        "sealed-gold.jsonl",
        "sealed-release.json",
    }
    for name in ("items.jsonl", "stores.jsonl", "sealed-gold.jsonl"):
        assert result.release_dir.joinpath(name).read_bytes() == source.joinpath(
            name
        ).read_bytes()
    assert (
        result.release_dir.joinpath("sealed-release.json").read_bytes()
        == result.manifest_bytes
    )
    assert stat.S_IMODE(result.release_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(
        result.release_dir.joinpath("items.jsonl").stat().st_mode
    ) == 0o644
    assert stat.S_IMODE(
        result.release_dir.joinpath("stores.jsonl").stat().st_mode
    ) == 0o644
    assert stat.S_IMODE(
        result.release_dir.joinpath("sealed-gold.jsonl").stat().st_mode
    ) == 0o600
    assert stat.S_IMODE(
        result.release_dir.joinpath("sealed-release.json").stat().st_mode
    ) == 0o644

    with pytest.raises(SealingError, match="already exists"):
        seal_release(
            source_dir=source,
            preregistration_path=preregistration,
            output_root=output_root,
            apply=True,
        )
    assert (
        result.release_dir.joinpath("sealed-release.json").read_bytes()
        == result.manifest_bytes
    )


def test_apply_rejects_conflicting_output_without_touching_it(tmp_path):
    source, preregistration = _write_source(tmp_path)
    output_root = tmp_path / "releases"
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    plan = seal_release(
        source_dir=source,
        preregistration_path=preregistration,
        output_root=output_root,
        apply=False,
    )
    plan.release_dir.mkdir(mode=0o700)
    sentinel = plan.release_dir / "other-owner"
    sentinel.write_text("untouched", encoding="utf-8")

    with pytest.raises(SealingError, match="already exists"):
        seal_release(
            source_dir=source,
            preregistration_path=preregistration,
            output_root=output_root,
            apply=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert sorted(path.name for path in output_root.iterdir()) == [plan.release_id]


def test_apply_removes_private_staging_after_mid_write_failure(
    tmp_path,
    monkeypatch,
):
    source, preregistration = _write_source(tmp_path)
    output_root = tmp_path / "releases"
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    original = sealing._write_file_at
    calls = 0

    def fail_third_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected staged write failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(sealing, "_write_file_at", fail_third_write)
    with pytest.raises(OSError, match="injected staged write failure"):
        seal_release(
            source_dir=source,
            preregistration_path=preregistration,
            output_root=output_root,
            apply=True,
        )

    assert list(output_root.iterdir()) == []


def test_apply_detects_source_toctou_before_publication(tmp_path, monkeypatch):
    source, preregistration = _write_source(tmp_path)
    output_root = tmp_path / "releases"
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    original = sealing._stage_release

    def mutate_after_staging(*args, **kwargs):
        staged = original(*args, **kwargs)
        source.joinpath("items.jsonl").write_bytes(b"changed\n")
        return staged

    monkeypatch.setattr(sealing, "_stage_release", mutate_after_staging)
    with pytest.raises(SealingError, match="changed"):
        seal_release(
            source_dir=source,
            preregistration_path=preregistration,
            output_root=output_root,
            apply=True,
        )

    assert list(output_root.iterdir()) == []


def test_publish_boundary_rejects_release_directory_swapped_before_return(
    tmp_path,
    monkeypatch,
):
    source, preregistration = _write_source(tmp_path)
    output_root = tmp_path / "releases"
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    plan = seal_release(
        source_dir=source,
        preregistration_path=preregistration,
        output_root=output_root,
        apply=False,
    )
    displaced = output_root / "displaced-published-release"
    sentinel = plan.release_dir / "replacement-sentinel"

    def swap_release(event, **_context):
        if event != "publish_before_final_return":
            return
        plan.release_dir.rename(displaced)
        plan.release_dir.mkdir(mode=0o700)
        plan.release_dir.chmod(0o700)
        sentinel.write_text("replacement", encoding="utf-8")

    monkeypatch.setattr(sealing, "_run_mutation_hook", swap_release)

    with pytest.raises(SealingError, match="changed|replaced|identity|entries"):
        seal_release(
            source_dir=source,
            preregistration_path=preregistration,
            output_root=output_root,
            apply=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "replacement"
    assert not displaced.exists()


def test_quarantine_cleanup_recursively_deletes_only_the_pinned_directory(
    tmp_path,
):
    parent = tmp_path / "cleanup-parent"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    owned = parent / "owned"
    owned.mkdir(mode=0o700)
    owned.chmod(0o700)
    owned.joinpath("payload").write_bytes(b"payload")
    nested = owned / "nested"
    nested.mkdir(mode=0o700)
    nested.chmod(0o700)
    nested.joinpath("child").write_bytes(b"child")
    parent_fd = os.open(parent, sealing._directory_flags())
    owned_fd = os.open("owned", sealing._directory_flags(), dir_fd=parent_fd)
    try:
        sealing._quarantine_and_delete_directory(
            parent_fd,
            "owned",
            owned_fd,
            label="test owned directory",
        )
    finally:
        os.close(owned_fd)
        os.close(parent_fd)

    assert list(parent.iterdir()) == []


def test_quarantine_cleanup_restores_replacement_directory_on_identity_mismatch(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "cleanup-parent"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    owned = parent / "owned"
    owned.mkdir(mode=0o700)
    owned.chmod(0o700)
    owned.joinpath("payload").write_bytes(b"exact-owned")
    parent_fd = os.open(parent, sealing._directory_flags())
    owned_fd = os.open("owned", sealing._directory_flags(), dir_fd=parent_fd)

    def replace_quarantine(event, **context):
        if (
            event != "cleanup_after_directory_quarantine"
            or context["source_name"] != "owned"
        ):
            return
        directory_fd = context["parent_fd"]
        quarantine_name = context["quarantine_name"]
        os.rename(
            quarantine_name,
            "displaced-exact-owned",
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.mkdir(quarantine_name, 0o700, dir_fd=directory_fd)
        replacement_fd = os.open(
            quarantine_name,
            sealing._directory_flags(),
            dir_fd=directory_fd,
        )
        try:
            _write_file_at(replacement_fd, "sentinel", b"replacement")
        finally:
            os.close(replacement_fd)

    monkeypatch.setattr(sealing, "_run_mutation_hook", replace_quarantine)
    try:
        with pytest.raises(SealingError, match="quarantine|identity|restore"):
            sealing._quarantine_and_delete_directory(
                parent_fd,
                "owned",
                owned_fd,
                label="test owned directory",
            )
    finally:
        os.close(owned_fd)
        os.close(parent_fd)

    assert parent.joinpath("owned", "sentinel").read_bytes() == b"replacement"
    assert (
        parent.joinpath("displaced-exact-owned", "payload").read_bytes()
        == b"exact-owned"
    )


def test_quarantine_cleanup_restores_replacement_member_on_identity_mismatch(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "cleanup-parent"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    owned = parent / "owned"
    owned.mkdir(mode=0o700)
    owned.chmod(0o700)
    owned.joinpath("payload").write_bytes(b"exact-owned")
    parent_fd = os.open(parent, sealing._directory_flags())
    owned_fd = os.open("owned", sealing._directory_flags(), dir_fd=parent_fd)

    def replace_member(event, **context):
        if (
            event != "cleanup_after_member_quarantine"
            or context["source_name"] != "payload"
        ):
            return
        directory_fd = context["parent_fd"]
        quarantine_name = context["quarantine_name"]
        os.rename(
            quarantine_name,
            "displaced-exact-member",
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        _write_file_at(directory_fd, quarantine_name, b"replacement")

    monkeypatch.setattr(sealing, "_run_mutation_hook", replace_member)
    try:
        with pytest.raises(SealingError, match="quarantine|identity|restore"):
            sealing._quarantine_and_delete_directory(
                parent_fd,
                "owned",
                owned_fd,
                label="test owned directory",
            )
    finally:
        os.close(owned_fd)
        os.close(parent_fd)

    assert parent.joinpath("owned", "payload").read_bytes() == b"replacement"
    assert (
        parent.joinpath("owned", "displaced-exact-member").read_bytes()
        == b"exact-owned"
    )


@pytest.mark.parametrize("mutation", ["insert", "remove", "replace"])
def test_quarantine_cleanup_detects_membership_mutation_after_rename(
    tmp_path,
    monkeypatch,
    mutation,
):
    parent = tmp_path / "cleanup-parent"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    owned = parent / "owned"
    owned.mkdir(mode=0o700)
    owned.chmod(0o700)
    owned.joinpath("payload").write_bytes(b"exact-owned")
    parent_fd = os.open(parent, sealing._directory_flags())
    owned_fd = os.open("owned", sealing._directory_flags(), dir_fd=parent_fd)

    def mutate_membership(event, **context):
        if (
            event != "cleanup_after_directory_quarantine"
            or context["source_name"] != "owned"
        ):
            return
        directory_fd = context["descriptor"]
        if mutation == "insert":
            _write_file_at(directory_fd, "unexpected", b"inserted")
        elif mutation == "remove":
            os.rename(
                "payload",
                "displaced-payload",
                src_dir_fd=directory_fd,
                dst_dir_fd=parent_fd,
            )
        else:
            os.rename(
                "payload",
                "displaced-payload",
                src_dir_fd=directory_fd,
                dst_dir_fd=parent_fd,
            )
            _write_file_at(directory_fd, "payload", b"replacement")

    monkeypatch.setattr(sealing, "_run_mutation_hook", mutate_membership)
    try:
        with pytest.raises(
            SealingError,
            match="changed|membership|identity|replaced",
        ):
            sealing._quarantine_and_delete_directory(
                parent_fd,
                "owned",
                owned_fd,
                label="test owned directory",
            )
    finally:
        os.close(owned_fd)
        os.close(parent_fd)

    assert owned.is_dir()
    if mutation == "insert":
        assert owned.joinpath("payload").read_bytes() == b"exact-owned"
        assert owned.joinpath("unexpected").read_bytes() == b"inserted"
    elif mutation == "remove":
        assert not owned.joinpath("payload").exists()
        assert parent.joinpath("displaced-payload").read_bytes() == b"exact-owned"
    else:
        assert owned.joinpath("payload").read_bytes() == b"replacement"
        assert parent.joinpath("displaced-payload").read_bytes() == b"exact-owned"


def test_cleanup_failure_still_closes_all_staged_descriptors(
    tmp_path,
    monkeypatch,
):
    source, preregistration = _write_source(tmp_path)
    output_root = tmp_path / "releases"
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    closed = False
    original_close = sealing._close_staged

    def mutate_published_membership(event, **context):
        if event == "publish_before_final_return":
            os.mkdir("unexpected", 0o700, dir_fd=context["release_fd"])

    def fail_cleanup(*_args, **_kwargs):
        raise SealingError("injected safe cleanup failure")

    def record_close(staged):
        nonlocal closed
        closed = True
        original_close(staged)

    monkeypatch.setattr(
        sealing,
        "_run_mutation_hook",
        mutate_published_membership,
    )
    monkeypatch.setattr(sealing, "_cleanup_staged_directory", fail_cleanup)
    monkeypatch.setattr(sealing, "_close_staged", record_close)

    with pytest.raises(SealingError, match="cleanup failure"):
        seal_release(
            source_dir=source,
            preregistration_path=preregistration,
            output_root=output_root,
            apply=True,
        )

    assert closed


@pytest.mark.parametrize("mutation", ["insert", "remove", "replace"])
def test_publish_boundary_detects_concurrent_membership_mutation(
    tmp_path,
    monkeypatch,
    mutation,
):
    source, preregistration = _write_source(tmp_path)
    output_root = tmp_path / "releases"
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    plan = seal_release(
        source_dir=source,
        preregistration_path=preregistration,
        output_root=output_root,
        apply=False,
    )
    replacement_bytes = source.joinpath("items.jsonl").read_bytes()

    def mutate_membership(event, **_context):
        if event != "publish_before_final_return":
            return
        if mutation == "insert":
            plan.release_dir.joinpath("unexpected").write_text(
                "inserted",
                encoding="utf-8",
            )
        elif mutation == "remove":
            plan.release_dir.joinpath("stores.jsonl").rename(
                output_root / "removed-published-stores.jsonl"
            )
        else:
            items = plan.release_dir / "items.jsonl"
            items.rename(output_root / "displaced-published-items.jsonl")
            items.write_bytes(replacement_bytes)
            items.chmod(0o644)

    monkeypatch.setattr(sealing, "_run_mutation_hook", mutate_membership)

    with pytest.raises(
        SealingError,
        match="changed|replaced|membership|entries",
    ):
        seal_release(
            source_dir=source,
            preregistration_path=preregistration,
            output_root=output_root,
            apply=True,
        )


@pytest.mark.parametrize("attack", ["symlink", "hardlink", "unsafe_mode"])
def test_sealing_rejects_unsafe_source_files(tmp_path, attack):
    source, preregistration = _write_source(tmp_path)
    items = source / "items.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(items.read_bytes())
    if attack == "symlink":
        items.unlink()
        items.symlink_to(replacement)
    elif attack == "hardlink":
        os.link(items, tmp_path / "items-hardlink.jsonl")
    else:
        items.chmod(0o666)

    with pytest.raises(SealingError, match="symlink|hard.link|mode|writable"):
        seal_release(
            source_dir=source,
            preregistration_path=preregistration,
            output_root=tmp_path / "releases",
            apply=False,
        )


def test_sealing_rejects_lexical_path_traversal_and_symlink_output(tmp_path):
    source, preregistration = _write_source(tmp_path)
    traversing_source = source / ".." / source.name

    with pytest.raises(SealingError, match="traversal"):
        seal_release(
            source_dir=traversing_source,
            preregistration_path=preregistration,
            output_root=tmp_path / "releases",
            apply=False,
        )
    with pytest.raises(SealingError, match="filesystem root|output root"):
        seal_release(
            source_dir=source,
            preregistration_path=preregistration,
            output_root=Path("/"),
            apply=False,
        )

    real_output = tmp_path / "real-output"
    real_output.mkdir(mode=0o700)
    symlink_output = tmp_path / "output-link"
    symlink_output.symlink_to(real_output, target_is_directory=True)
    with pytest.raises(SealingError, match="symlink"):
        seal_release(
            source_dir=source,
            preregistration_path=preregistration,
            output_root=symlink_output,
            apply=False,
        )
    with pytest.raises(SealingError, match="symlink"):
        seal_release(
            source_dir=source,
            preregistration_path=preregistration,
            output_root=symlink_output,
            apply=True,
        )


def test_explicit_verification_replays_every_record_and_external_commitment(
    tmp_path,
):
    source, preregistration = _write_source(tmp_path)
    output_root = tmp_path / "releases"
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    published = seal_release(
        source_dir=source,
        preregistration_path=preregistration,
        output_root=output_root,
        apply=True,
    )

    verified = verify_release(
        release_dir=published.release_dir,
        expected_release_sha256=published.release_sha256,
    )

    assert verified.release_dir == published.release_dir
    assert verified.release_sha256 == published.release_sha256
    assert verified.manifest_bytes == published.manifest_bytes
    assert verified.item_count == published.item_count
    assert verified.pair_count == published.pair_count
    assert verified.world_count == published.world_count
    assert verified.store_count == published.store_count
    assert verified.sealed_gold_sha256 == json.loads(
        published.manifest_bytes
    )["sealed_gold"]["sha256"]

    with pytest.raises(SealingError, match="external|commitment"):
        verify_release(
            release_dir=published.release_dir,
            expected_release_sha256="0" * 64,
        )


def test_model_visible_preflight_never_opens_or_returns_sealed_gold(
    tmp_path,
    monkeypatch,
):
    source, preregistration = _write_source(tmp_path)
    output_root = tmp_path / "releases"
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    published = seal_release(
        source_dir=source,
        preregistration_path=preregistration,
        output_root=output_root,
        apply=True,
    )
    opened: list[str] = []
    original = sealing._open_pinned_file

    def reject_gold(parent_fd, name, label):
        opened.append(name)
        if name == "sealed-gold.jsonl":
            raise AssertionError("model-visible preflight opened sealed gold")
        return original(parent_fd, name, label)

    monkeypatch.setattr(sealing, "_open_pinned_file", reject_gold)
    preflight = preflight_model_visible_release(
        release_dir=published.release_dir,
        expected_release_sha256=published.release_sha256,
    )

    assert opened == ["sealed-release.json", "items.jsonl", "stores.jsonl"]
    assert preflight.release_sha256 == published.release_sha256
    assert preflight.item_count == published.item_count
    assert preflight.store_count == published.store_count
    assert preflight.sealed_gold_sha256 == json.loads(
        published.manifest_bytes
    )["sealed_gold"]["sha256"]
    assert not hasattr(preflight, "gold")
    assert not hasattr(preflight, "gold_content")
    assert not hasattr(preflight, "gold_records")


def test_model_visible_preflight_rejects_unsafe_gold_metadata_without_reading_it(
    tmp_path,
):
    source, preregistration = _write_source(tmp_path)
    output_root = tmp_path / "releases"
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    published = seal_release(
        source_dir=source,
        preregistration_path=preregistration,
        output_root=output_root,
        apply=True,
    )
    gold = published.release_dir / "sealed-gold.jsonl"
    gold.unlink()
    gold.symlink_to(source / "sealed-gold.jsonl")

    with pytest.raises(SealingError, match="symlink|regular"):
        preflight_model_visible_release(
            release_dir=published.release_dir,
            expected_release_sha256=published.release_sha256,
        )


@pytest.mark.parametrize(
    ("operation_name", "operation"),
    [
        (
            "preflight",
            lambda published: preflight_model_visible_release(
                release_dir=published.release_dir,
                expected_release_sha256=published.release_sha256,
            ),
        ),
        (
            "verify",
            lambda published: verify_release(
                release_dir=published.release_dir,
                expected_release_sha256=published.release_sha256,
            ),
        ),
    ],
)
def test_read_boundary_rejects_release_directory_swapped_before_return(
    tmp_path,
    monkeypatch,
    operation_name,
    operation,
):
    _source, _preregistration, _output_root, published = _publish_fixture(
        tmp_path
    )
    displaced = tmp_path / f"{operation_name}-displaced-release"
    sentinel = published.release_dir / "replacement-sentinel"

    def swap_release(event, **_context):
        if event != f"{operation_name}_before_final_return":
            return
        published.release_dir.rename(displaced)
        published.release_dir.mkdir(mode=0o700)
        published.release_dir.chmod(0o700)
        sentinel.write_text("replacement", encoding="utf-8")

    monkeypatch.setattr(
        sealing,
        "_run_mutation_hook",
        swap_release,
        raising=False,
    )

    with pytest.raises(SealingError, match="changed|replaced|identity|entries"):
        operation(published)

    assert sentinel.read_text(encoding="utf-8") == "replacement"
    assert displaced.is_dir()


@pytest.mark.parametrize("operation_name", ["preflight", "verify"])
@pytest.mark.parametrize("mutation", ["insert", "remove", "replace"])
def test_read_boundary_detects_concurrent_membership_mutation(
    tmp_path,
    monkeypatch,
    operation_name,
    mutation,
):
    _source, _preregistration, _output_root, published = _publish_fixture(
        tmp_path
    )
    replacement_bytes = published.release_dir.joinpath("items.jsonl").read_bytes()

    def mutate_membership(event, **_context):
        if event != f"{operation_name}_before_final_return":
            return
        if mutation == "insert":
            published.release_dir.joinpath("unexpected").write_text(
                "inserted",
                encoding="utf-8",
            )
        elif mutation == "remove":
            published.release_dir.joinpath("stores.jsonl").rename(
                tmp_path / f"{operation_name}-removed-stores.jsonl"
            )
        else:
            items = published.release_dir / "items.jsonl"
            items.rename(tmp_path / f"{operation_name}-displaced-items.jsonl")
            items.write_bytes(replacement_bytes)
            items.chmod(0o644)

    monkeypatch.setattr(
        sealing,
        "_run_mutation_hook",
        mutate_membership,
        raising=False,
    )
    operation = (
        preflight_model_visible_release
        if operation_name == "preflight"
        else verify_release
    )

    with pytest.raises(
        SealingError,
        match="changed|replaced|membership|entries",
    ):
        operation(
            release_dir=published.release_dir,
            expected_release_sha256=published.release_sha256,
        )


@pytest.mark.parametrize("artifact", ["items.jsonl", "sealed-gold.jsonl"])
def test_explicit_verification_rejects_artifact_tampering(tmp_path, artifact):
    source, preregistration = _write_source(tmp_path)
    output_root = tmp_path / "releases"
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    published = seal_release(
        source_dir=source,
        preregistration_path=preregistration,
        output_root=output_root,
        apply=True,
    )
    published.release_dir.joinpath(artifact).write_bytes(b"tampered\n")

    with pytest.raises(SealingError, match="hash|commitment|canonical|gold"):
        verify_release(
            release_dir=published.release_dir,
            expected_release_sha256=published.release_sha256,
        )


def test_explicit_verification_rejects_hardlinks_and_wrong_content_address(
    tmp_path,
):
    source, preregistration = _write_source(tmp_path)
    output_root = tmp_path / "releases"
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    published = seal_release(
        source_dir=source,
        preregistration_path=preregistration,
        output_root=output_root,
        apply=True,
    )
    items = published.release_dir / "items.jsonl"
    os.link(items, tmp_path / "release-items-hardlink.jsonl")
    with pytest.raises(SealingError, match="hard.link"):
        verify_release(
            release_dir=published.release_dir,
            expected_release_sha256=published.release_sha256,
        )

    (tmp_path / "release-items-hardlink.jsonl").unlink()
    wrong_name = output_root / "not-content-addressed"
    published.release_dir.rename(wrong_name)
    with pytest.raises(SealingError, match="content.address"):
        verify_release(
            release_dir=wrong_name,
            expected_release_sha256=published.release_sha256,
        )


def test_cli_dry_run_publish_and_verify_each_emit_one_json_result(
    tmp_path,
    capsys,
):
    source, preregistration = _write_source(tmp_path)
    output_root = tmp_path / "releases"
    build_args = [
        "--source-dir",
        str(source),
        "--preregistration",
        str(preregistration),
        "--out-dir",
        str(output_root),
    ]

    assert sealing_cli.main([*build_args, "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    dry_run = json.loads(captured.out)
    assert dry_run["ok"] is True
    assert dry_run["mode"] == "dry-run"
    assert dry_run["published"] is False
    assert not output_root.exists()

    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    assert sealing_cli.main([*build_args, "--apply"]) == 0
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    published = json.loads(captured.out)
    assert published["ok"] is True
    assert published["mode"] == "publish"
    assert published["published"] is True
    assert Path(published["release_dir"]).is_dir()

    assert (
        sealing_cli.main(
            [
                "--verify-release",
                published["release_dir"],
                "--expected-release-sha256",
                published["release_sha256"],
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    verified = json.loads(captured.out)
    assert verified["ok"] is True
    assert verified["mode"] == "verify"
    assert verified["verified"] is True
    assert verified["published"] is False
    assert verified["release_sha256"] == published["release_sha256"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["--apply"],
        ["--verify-release", "missing"],
        ["--unknown"],
    ],
)
def test_cli_help_and_errors_preserve_json_only_stdout(arguments, capsys):
    return_code = sealing_cli.main(arguments)

    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    result = json.loads(captured.out)
    if arguments == ["--help"]:
        assert return_code == 0
        assert result["ok"] is True
        assert result["mode"] == "help"
        assert "usage" in result
    else:
        assert return_code != 0
        assert result["ok"] is False
        assert result["published"] is False
        assert result["error"]["code"]
        assert result["error"]["message"]


def test_cli_emits_repository_canonical_utf8_and_rejects_nonfinite(
    tmp_path,
    capsys,
):
    value = {"message": "mémoire 雪"}
    sealing_cli._emit(value)
    captured = capsys.readouterr()
    assert captured.out == canonical_json_bytes(value).decode("utf-8")
    assert "mémoire 雪" in captured.out
    assert "\\u" not in captured.out

    with pytest.raises(ValueError, match="non-canonical|non-finite"):
        sealing_cli._emit({"bad": float("nan")})
    assert capsys.readouterr().out == ""

    unicode_root = tmp_path / "资料"
    unicode_root.mkdir(mode=0o700)
    source, preregistration = _write_source(unicode_root)
    output_root = unicode_root / "发布"
    return_code = sealing_cli.main(
        [
            "--source-dir",
            str(source),
            "--preregistration",
            str(preregistration),
            "--out-dir",
            str(output_root),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert return_code == 0
    assert "资料" in captured.out
    assert "发布" in captured.out
    assert captured.out == canonical_json_bytes(
        json.loads(captured.out)
    ).decode("utf-8")
