import hashlib
import inspect
import json
from pathlib import Path

import pytest

import corpusgen.relation_schema as relation_schema_module
from corpusgen.relation_schema import (
    LITERAL_RELATIONS,
    InstrumentError,
    RelationSchema,
    RelationSpec,
    RelationStats,
    build_relation_schema,
    compute_relation_stats,
    select_entity_relations,
)


EXPECTED_LITERALS = (
    ("SYN_L0", ("birth date", "date of birth"), "date"),
    ("SYN_L1", ("founding date", "date founded"), "date"),
    ("SYN_L2", ("population", "number of residents"), "quantity"),
    ("SYN_L3", ("duration", "length of time"), "quantity"),
    ("SYN_L4", ("category alpha", "first category"), "category"),
    ("SYN_L5", ("category beta", "second category"), "category"),
    ("SYN_L6", ("reference code", "identifier code"), "string"),
    ("SYN_L7", ("short label", "display label"), "string"),
)


def _valid_stats(count):
    return [
        RelationStats(
            f"P{index}",
            5_000,
            4_750,
            4_000,
            10_000,
            (f"relation {index}",),
        )
        for index in range(1, count + 1)
    ]


def _write_training_fixture(directory):
    path = directory / "wikidata5m_transductive_train.txt"
    rows = [
        f"Q{subject}\tP2\tQ{subject + 10_000}\n"
        for subject in range(1, 4_751)
    ]
    rows.extend(
        f"Q{subject}\tP2\tQ{subject + 20_000}\n"
        for subject in range(1, 251)
    )
    rows.extend(
        [
            "Q1\tP10\tQ2\n",
            "Q2\tP10\tQ3\n",
            "Q3\tP10\tQ4\n",
            "Q4\tP10\tQ5\n",
            "Q5\tP10\tQ6\n",
        ]
    )
    path.write_text("".join(rows))
    return path


def _stub_relation_stats(monkeypatch, stats):
    monkeypatch.setattr(
        relation_schema_module,
        "compute_relation_stats",
        lambda transductive_train, aliases: tuple(stats),
    )


def _minimal_schema_manifest():
    return RelationSchema(
        catalog=(
            RelationSpec(
                relation_id="P1",
                aliases=("relation one",),
                target_kind="entity",
                support=10,
                distinct_subjects=10,
                distinct_objects=5,
                entity_count=10,
            ),
            RelationSpec(
                relation_id="SYN_L0",
                aliases=("birth date",),
                target_kind="date",
            ),
        ),
        path_relation_ids=("P1", "SYN_L0"),
    ).to_dict()


def _v1_schema_manifest():
    manifest = _minimal_schema_manifest()
    manifest["version"] = 1
    del manifest["entity_count"]
    for spec in manifest["catalog"]:
        spec.pop("distinct_objects")
        spec.pop("entity_count")
    return manifest


def test_literal_relation_specs_are_frozen_verbatim():
    assert LITERAL_RELATIONS == EXPECTED_LITERALS


@pytest.mark.parametrize("version", [True, 2.0, "2", 1])
def test_schema_artifacts_reject_unsupported_or_non_integer_versions(version):
    manifest = _minimal_schema_manifest()
    manifest["version"] = version

    with pytest.raises(ValueError, match="unsupported relation schema version"):
        RelationSchema.from_dict(manifest)


def test_genuine_v1_schema_is_rejected_before_v2_field_validation():
    with pytest.raises(ValueError, match="unsupported relation schema version"):
        RelationSchema.from_dict(_v1_schema_manifest())


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_schema_artifact_field_sets_are_validated_separately(mutation):
    manifest = _minimal_schema_manifest()
    if mutation == "missing":
        del manifest["catalog"]
    else:
        manifest["unexpected"] = None

    with pytest.raises(ValueError, match="invalid relation schema fields"):
        RelationSchema.from_dict(manifest)


@pytest.mark.parametrize(
    "relation_id",
    [relation_id for relation_id, _, _ in LITERAL_RELATIONS],
)
def test_literal_relation_artifacts_reject_complete_statistics(relation_id):
    spec = next(
        RelationSpec(relation_id, aliases, target_kind)
        for candidate, aliases, target_kind in LITERAL_RELATIONS
        if candidate == relation_id
    ).to_dict()
    spec.update(
        support=10,
        distinct_subjects=10,
        distinct_objects=5,
        entity_count=10,
    )

    with pytest.raises(
        ValueError,
        match="literal relations must not carry statistics",
    ):
        RelationSpec.from_dict(spec)


@pytest.mark.parametrize(
    "statistic",
    ["support", "distinct_subjects", "distinct_objects", "entity_count"],
)
def test_literal_relation_artifacts_reject_each_non_null_statistic(statistic):
    spec = RelationSpec(
        "SYN_L0",
        ("birth date",),
        "date",
    ).to_dict()
    spec[statistic] = 0

    with pytest.raises(
        ValueError,
        match="literal relations must not carry statistics",
    ):
        RelationSpec.from_dict(spec)


def test_selector_uses_support_functionality_and_stable_tiebreak():
    stats = [
        RelationStats("P3", 6_000, 5_900, 5_000, 10_000, ("c",)),
        RelationStats("P2", 6_000, 5_900, 5_000, 10_000, ("b",)),
        RelationStats("P1", 7_000, 6_650, 5_000, 10_000, ("a",)),
    ]

    selected = select_entity_relations(stats, count=3)

    assert [item.relation_id for item in selected] == ["P1", "P2", "P3"]
    assert selected[0].functionality == 0.95


def test_selector_applies_frozen_alias_support_and_functionality_thresholds():
    stats = [
        RelationStats(
            "P1", 5_000, 4_750, 4_000, 10_000, ("survives",)
        ),
        RelationStats(
            "P2", 4_999, 4_999, 4_000, 10_000, ("low support",)
        ),
        RelationStats(
            "P3", 5_000, 4_749, 4_000, 10_000, ("low functionality",)
        ),
        RelationStats("P4", 9_000, 9_000, 8_000, 10_000, ()),
    ]

    assert select_entity_relations(stats, count=1) == (stats[0],)


def test_selector_fails_when_fewer_than_thirty_two_survive():
    with pytest.raises(InstrumentError, match="fewer than 32"):
        select_entity_relations(_valid_stats(31), count=32)


def test_relation_stats_count_only_transductive_training_with_sqlite(tmp_path):
    train = _write_training_fixture(tmp_path)
    # If the builder globbed adjacent valid/test files, P10 would outrank P2.
    for split in ("valid", "test"):
        (tmp_path / f"wikidata5m_inductive_{split}.txt").write_text(
            "".join(
                f"Q{subject}\tP10\tQ{subject + 30_000}\n"
                for subject in range(1, 6_001)
            )
        )
    aliases = {
        "P10": ("relation ten",),
        "P2": ("relation two",),
        "P3": ("shared",),
        "P4": (" Shared ",),
    }

    stats = compute_relation_stats(train, aliases)

    by_id = {item.relation_id: item for item in stats}
    assert by_id["P2"].support == 5_000
    assert by_id["P2"].distinct_subjects == 4_750
    assert by_id["P2"].distinct_objects == 5_000
    assert by_id["P2"].entity_count == 9_750
    assert by_id["P2"].functionality == 0.95
    assert by_id["P2"].subject_coverage == 4_750 / 9_750
    assert by_id["P2"].target_pool_ratio == 5_000 / 4_750
    assert by_id["P10"].support == 5
    assert by_id["P10"].distinct_subjects == 5
    assert by_id["P10"].distinct_objects == 5
    assert by_id["P10"].entity_count == 9_750


def test_relation_stats_cleanup_uses_work_root_and_leaves_it_empty(tmp_path):
    train = _write_training_fixture(tmp_path)
    aliases = {
        "P10": ("relation ten",),
        "P2": ("relation two",),
        "P3": ("shared",),
        "P4": (" Shared ",),
    }
    work_root = tmp_path / "sqlite-work"
    work_root.mkdir()
    stats = compute_relation_stats(train, aliases, work_root=work_root)
    assert stats
    assert list(work_root.iterdir()) == []


def test_public_builder_rejects_cardinality_override(monkeypatch, tmp_path):
    stats = _valid_stats(32)
    _stub_relation_stats(monkeypatch, stats)
    train = tmp_path / "wikidata5m_transductive_train.txt"
    aliases = {item.relation_id: item.aliases for item in stats}

    with pytest.raises(TypeError, match="unexpected keyword argument 'count'"):
        build_relation_schema(train, aliases, count=1)

    assert "count" not in inspect.signature(build_relation_schema).parameters


def test_public_builder_fails_closed_with_thirty_one_survivors(
    monkeypatch, tmp_path
):
    stats = _valid_stats(31)
    _stub_relation_stats(monkeypatch, stats)
    train = tmp_path / "wikidata5m_transductive_train.txt"
    aliases = {item.relation_id: item.aliases for item in stats}

    with pytest.raises(InstrumentError, match="fewer than 32"):
        build_relation_schema(train, aliases)


def test_public_builder_returns_exactly_thirty_two_entity_relations(
    monkeypatch, tmp_path
):
    stats = _valid_stats(33)
    _stub_relation_stats(monkeypatch, stats)
    train = tmp_path / "wikidata5m_transductive_train.txt"
    aliases = {item.relation_id: item.aliases for item in stats}

    schema = build_relation_schema(train, aliases)

    entity_path = [
        spec.relation_id
        for spec in schema.path_relations
        if spec.target_kind == "entity"
    ]
    assert entity_path == [f"P{index}" for index in range(1, 33)]
    assert len(schema.path_relation_ids) == 40


def test_build_schema_refuses_non_transductive_input(tmp_path):
    path = tmp_path / "wikidata5m_inductive_valid.txt"
    path.write_text("Q1\tP1\tQ2\n")

    with pytest.raises(ValueError, match="transductive training"):
        build_relation_schema(path, {"P1": ("one",)})


def test_schema_manifest_is_canonical_and_independent_of_input_order(
    monkeypatch, tmp_path
):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    stats = _valid_stats(32)

    def fake_stats(transductive_train, aliases):
        values = stats if Path(transductive_train).parent == first_dir else reversed(stats)
        return tuple(values)

    monkeypatch.setattr(
        relation_schema_module,
        "compute_relation_stats",
        fake_stats,
    )
    first = build_relation_schema(
        first_dir / "wikidata5m_transductive_train.txt",
        {
            item.relation_id: item.aliases
            for item in reversed(stats)
        },
    )
    second = build_relation_schema(
        second_dir / "wikidata5m_transductive_train.txt",
        {item.relation_id: item.aliases for item in stats},
    )

    assert first == second
    assert first.to_dict()["version"] == 2
    assert first.entity_count == 10_000
    assert first.to_dict()["entity_count"] == 10_000
    assert all(
        spec.distinct_objects is not None and spec.entity_count is not None
        for spec in first.catalog
        if spec.target_kind == "entity"
    )
    assert all(
        spec.support is None
        and spec.distinct_subjects is None
        and spec.distinct_objects is None
        and spec.entity_count is None
        for spec in first.catalog
        if spec.target_kind != "entity"
    )
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.canonical_bytes() == (
        json.dumps(
            first.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    assert first.sha256() == hashlib.sha256(first.canonical_bytes()).hexdigest()

    path = tmp_path / "relation-schema.json"
    first.write(path)
    assert path.read_bytes() == first.canonical_bytes()
    assert RelationSchema.from_path(path) == first
