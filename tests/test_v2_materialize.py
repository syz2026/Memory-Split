import hashlib
import io
import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import pytest

import corpusgen.parallel.production as production_module
import corpusgen.v2_materialize as materialize_module
import corpusgen.v2_objective as objective_module
from corpusgen.graph_trace import parse_serialized_action
from corpusgen.parallel import FROZEN_LANES
from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.reasoning.proofs import (
    GraphTraversalPremise,
    solve_graph_traversal,
    verify_proof,
)
from corpusgen.v2_materialize import (
    _GRAPH_WORLD_OFFSET,
    _MULTIHOP_WORLD_OFFSET,
    _REFINEMENT_WORLD_OFFSET,
    LaneRecord,
    LaneSegment,
    V2MaterializationError,
    _compile_exact_lane,
    _encode_record,
    _generated_lock_payloads,
    _LaneWriter,
    _ObjectiveSource,
    _ParquetTextSource,
    _RecordSource,
    _RelationalRefinementSource,
    _solver_bundle,
    _subset_indices_exact,
    _SyntheticGraphSource,
    _SyntheticMultihopSource,
    _WikidataGraphSource,
    _WikidataIndex,
    _WikidataPathSource,
    materialization_status,
    materialize_v2_smoke,
)
from corpusgen.v2_objective import _ProntoQA, _seed
from train.tokenizer import get_tok


def _objective_verification(answer: str) -> dict:
    return {
        "answer": answer,
        "kind": "objective_answer",
        "reference_answer": answer,
        "validator": "canonical_exact_match",
    }


def _source_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_graph_traversal_solver_replays_only_contiguous_paths():
    premises = (
        GraphTraversalPremise("edge-b", 1, "Q2", "P2", "Q3"),
        GraphTraversalPremise("edge-a", 0, "Q1", "P1", "Q2"),
    )

    proof = solve_graph_traversal(premises)

    assert proof.family == "graph_path_traversal"
    assert proof.conclusion == (("endpoint", "Q3"),)
    assert proof.premise_ids == ("edge-a", "edge-b")
    assert verify_proof(proof, premises) is True
    with pytest.raises(ValueError, match="contiguous path"):
        solve_graph_traversal(
            (
                premises[1],
                GraphTraversalPremise("edge-c", 1, "Q9", "P2", "Q3"),
            )
        )


def test_exact_subset_selection_is_stable_and_source_ordered():
    assert _subset_indices_exact((7, 4, 5, 3), 12) == (0, 2)
    assert _subset_indices_exact((4, 6), 5) is None
    assert _subset_indices_exact((4, 6), 0) == ()


def test_exact_subset_selection_matches_bounded_brute_force():
    from itertools import combinations, product

    for size in range(5):
        for lengths in product(range(1, 5), repeat=size):
            possible = {
                sum(lengths[index] for index in selected)
                for count in range(size + 1)
                for selected in combinations(range(size), count)
            }
            for target in range(13):
                selected = _subset_indices_exact(lengths, target)
                assert (selected is not None) is (target in possible)
                if selected is not None:
                    assert tuple(sorted(set(selected))) == selected
                    assert sum(lengths[index] for index in selected) == target


def test_parquet_source_backs_off_a_lossy_utf8_token_boundary(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    tok = get_tok()
    prefix = " a" * 2_045
    text = prefix + "ࡀ"
    assert len(tok.encode(text)) == 2_048
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"text": [text]}), path)
    source = _ParquetTextSource(
        ((path, None),),
        source_id="fineweb_edu",
        record_prefix="fixture",
    )

    first, cursor = source.next({})
    second, final_cursor = source.next(cursor)
    source.close()

    assert first.segments[0].text == prefix
    assert second.segments[0].text == "ࡀ"
    assert tok.encode(first.segments[0].text + second.segments[0].text) == tok.encode(
        text
    )
    assert cursor["token_offset"] == 2_045
    assert final_cursor["row"] == 1


def test_objective_seed_schedule_is_collision_free_over_record_attempts():
    seeds = {
        _seed("deepmind_mathematics_generator", index, attempt)
        for index in range(2_000)
        for attempt in range(4)
    }
    assert len(seeds) == 8_000


def test_objective_worker_disables_bytecode_and_uses_scratch_cwd(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "immutable-source"
    source.mkdir()
    log_root = tmp_path / "work" / "logs"
    captured = {}

    class _Process:
        def __init__(self):
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(
                json.dumps(
                    {
                        "format": materialize_module.WORKER_FORMAT,
                        "provider": "fixture",
                        "ready": True,
                        "runtime": {"python": "fixture"},
                    }
                ).encode()
                + b"\n"
            )

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout):
            return 0

    def popen(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return _Process()

    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    monkeypatch.setattr(materialize_module.subprocess, "Popen", popen)
    client = materialize_module._ObjectiveWorkerClient(
        "fixture",
        source,
        python=Path(sys.executable),
        log_root=log_root,
    )
    client.close()

    assert captured["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert captured["cwd"] == log_root / "workers" / "fixture"
    assert captured["cwd"] != source
    assert list(source.iterdir()) == []


def test_objective_worker_rejects_scratch_inside_staged_source(tmp_path):
    source = tmp_path / "immutable-source"
    source.mkdir()

    with pytest.raises(V2MaterializationError) as captured:
        materialize_module._ObjectiveWorkerClient(
            "fixture",
            source,
            python=Path(sys.executable),
            log_root=source / "work",
        )

    assert captured.value.code == "objective_work_root_overlaps_source"
    assert list(source.iterdir()) == []


def test_materializer_rejects_work_root_inside_source_stage(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    work = stage / "work"

    with pytest.raises(V2MaterializationError) as captured:
        materialize_module.materialize_v2_source_root(
            stage,
            tmp_path / "published",
            work,
        )

    assert captured.value.code == "work_root_overlaps_source_stage"
    assert not work.exists()


def test_prontoqa_import_reads_locked_relative_resource_from_scratch(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "prontoqa"
    source.mkdir()
    (source / "bad_patterns.txt").write_text("locked pattern\n", encoding="utf-8")
    (source / "run_experiment.py").write_text(
        "from pathlib import Path\n"
        "RESOURCE = Path('bad_patterns.txt').read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    original_path = list(sys.path)
    monkeypatch.chdir(scratch)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.delitem(sys.modules, "run_experiment", raising=False)
    try:
        provider = objective_module._ProntoQA(source)
    finally:
        sys.path[:] = original_path
        sys.modules.pop("run_experiment", None)

    assert provider._module.RESOURCE == "locked pattern\n"
    assert Path.cwd() == scratch
    assert not (source / "__pycache__").exists()


def test_prontoqa_rejection_sampling_resets_state_and_rotates_depth():
    calls = []

    class _Config:
        def __init__(self, **values):
            self.values = values

    class _Module:
        OntologyConfig = _Config
        config = None

        def generate_question(self, steps, *_args, **_kwargs):
            calls.append((steps, self.config))
            if len(calls) == 1:
                return (None,) * 6
            return (
                "premise text",
                "query text",
                (),
                ["proof step"],
                True,
                (),
            )

    provider = object.__new__(_ProntoQA)
    provider._module = _Module()

    record = provider.generate(0)

    assert [steps for steps, _config in calls] == [2, 3]
    assert calls[0][1] is not calls[1][1]
    assert record["metadata"]["attempt"] == 1
    assert record["metadata"]["deduction_steps"] == 3
    assert len(record["metadata"]["native_proof_trace_sha256"]) == 64


def test_prontoqa_exhausts_primary_before_rotating_depth_pairing():
    calls = []

    class _Config:
        def __init__(self, **values):
            self.values = values

    class _Module:
        OntologyConfig = _Config
        config = None

        def generate_question(self, steps, *_args, **_kwargs):
            calls.append((steps, self.config))
            if len(calls) <= 160:
                return (None,) * 6
            return (
                "Alex is a gorpus. Every gorpus is not earthy.",
                "Alex is earthy.",
                (),
                [
                    "Alex is a gorpus.",
                    "Every gorpus is not earthy.",
                    "Alex is not earthy.",
                ],
                False,
                (),
            )

    provider = object.__new__(_ProntoQA)
    provider._module = _Module()

    record = provider.generate(2317)

    assert [steps for steps, _config in calls[:128]] == [
        2 + ((2317 + attempt) % 5) for attempt in range(128)
    ]
    assert [steps for steps, _config in calls[128:]] == [
        2 + ((2317 + attempt + 1) % 5) for attempt in range(33)
    ]
    assert len({id(config) for _steps, config in calls}) == len(calls)
    assert record["metadata"] == {
        "attempt": 32,
        "deduction_steps": 2,
        "depth_schedule_pass": 1,
        "native_proof_trace_sha256": hashlib.sha256(
            canonical_json_bytes(
                [
                    "Alex is a gorpus.",
                    "Every gorpus is not earthy.",
                    "Alex is not earthy.",
                ]
            )
        ).hexdigest(),
        "rejected_native_samples": 160,
        "rejection_policy": "exhaust_primary_then_rotate_seed_depth_pairing",
        "seed": _seed("prontoqa", 2317, 32),
    }


def test_prontoqa_fails_closed_after_all_seed_depth_combinations():
    calls = []

    class _Config:
        def __init__(self, **values):
            self.values = values

    class _Module:
        OntologyConfig = _Config
        config = None

        def generate_question(self, steps, *_args, **_kwargs):
            calls.append((steps, self.config))
            return (None,) * 6

    provider = object.__new__(_ProntoQA)
    provider._module = _Module()

    with pytest.raises(RuntimeError, match="640 frozen seed/depth combinations"):
        provider.generate(2317)

    assert len(calls) == 640
    assert len({id(config) for _steps, config in calls}) == len(calls)
    for attempt in range(128):
        assert {
            calls[depth_pass * 128 + attempt][0] for depth_pass in range(5)
        } == {2, 3, 4, 5, 6}


def test_objective_prefetch_preserves_serial_record_and_cursor_order():
    providers = tuple(f"provider-{index}" for index in range(5))

    class _Client:
        def __init__(self, provider):
            self.provider = provider
            self.calls = []
            self.runtime = {"python": "fixture"}

        def generate(self, index):
            self.calls.append(index)
            return {
                "answer": f"answer-{self.provider}-{index}",
                "metadata": {"provider": self.provider},
                "question": f"question-{self.provider}-{index}",
            }

        def close(self):
            return None

    def _source(prefetch):
        source = _ObjectiveSource.__new__(_ObjectiveSource)
        source.clients = {provider: _Client(provider) for provider in providers}
        source.puzzles = {}
        source.providers = providers
        source.tok = get_tok()
        source.runtime = {
            provider: client.runtime for provider, client in source.clients.items()
        }
        source.expected_runtime = {
            provider: {"probe_record_sha256s": []} for provider in providers
        }
        source._executor = (
            ThreadPoolExecutor(max_workers=len(providers)) if prefetch else None
        )
        source._pending = {}
        return source

    serial = _source(False)
    parallel = _source(True)
    serial_cursor = {}
    parallel_cursor = {}
    try:
        for _ in range(30):
            serial_record, serial_cursor = serial.next(serial_cursor)
            parallel_record, parallel_cursor = parallel.next(parallel_cursor)
            assert parallel_record == serial_record
            assert parallel_cursor == serial_cursor
    finally:
        serial.close()
        parallel.close()

    assert {
        provider: client.calls for provider, client in serial.clients.items()
    } == {
        provider: client.calls[:6]
        for provider, client in parallel.clients.items()
    }


def test_deepmind_adapter_removes_object_hash_order_before_seeded_shuffle(
    monkeypatch,
):
    class _Entity:
        def __init__(self, name):
            self.child_description = f"child-{name}"
            self.description = f"description-{name}"
            self.expression_used = False
            self.handle = f"handle-{name}"

    class _Composition:
        Entity = _Entity

    class _Context:
        def __init__(self, children):
            self.child_entities = children

    first = _Entity("first")
    second = _Entity("second")
    shuffle_inputs = []

    def reverse(values):
        shuffle_inputs.append(tuple(item.handle for item in values))
        values.reverse()

    monkeypatch.setattr(objective_module.random, "shuffle", reverse)
    description, expanded = objective_module._deepmind_expand_entities(
        _Composition,
        _Context([first, second, first]),
        literal="unchanged",
        selected=second,
    )

    assert shuffle_inputs == [("handle-first", "handle-second")]
    assert expanded == {
        "literal": "unchanged",
        "selected": "handle-second",
    }
    assert description == (
        "child-second description-second child-first description-first"
    )
    assert objective_module._deepmind_randint(2.0, 2.0) == 2
    with pytest.raises(TypeError):
        objective_module._deepmind_randint(1.5, 2.5)


def test_deepmind_retries_only_the_zero_term_entropy_assertion():
    calls = 0

    def integers_with_sum(value, count, entropy):
        assert count > 0 or value != 0 or entropy <= 0

    class _SympyRandom:
        @staticmethod
        def seed(_value):
            return None

    class _Problem:
        answer = "-36"
        question = "Differentiate the fixture polynomial."

    def module():
        nonlocal calls
        calls += 1
        if calls == 1:
            integers_with_sum(0, 0, 1.0)
        return _Problem()

    generator = object.__new__(objective_module._DeepMindMathematics)
    generator._integers_with_sum = integers_with_sum
    generator._modules = (("calculus__fixture", module),)
    generator._sympy_random = _SympyRandom()

    record = generator.generate(295_194)

    assert calls == 2
    assert record["answer"] == "-36"
    assert record["metadata"] == {
        "attempt": 1,
        "module": "calculus__fixture",
        "native_assertion_rejections": 1,
        "rejected_native_samples": 1,
        "rejection_policy": "zero_term_nonzero_entropy_assertion",
        "seed": _seed("deepmind_mathematics_generator", 295_194, 1),
    }

    def unrelated_assertion():
        raise AssertionError("unrelated")

    generator._modules = (("calculus__fixture", unrelated_assertion),)
    with pytest.raises(AssertionError, match="unrelated"):
        generator.generate(295_194)


def test_lane_checkpoint_resumes_and_rejects_verified_record_reuse(tmp_path):
    tok = get_tok()
    records = (
        LaneRecord(
            "objective-0",
            "objective",
            (LaneSegment("first objective record"),),
            _objective_verification("first"),
        ),
        LaneRecord(
            "objective-1",
            "objective",
            (LaneSegment("second objective record"),),
            _objective_verification("second"),
        ),
    )
    encoded = tuple(_encode_record(tok, record) for record in records)
    quota = sum(record.token_count for record in encoded)

    writer = _LaneWriter(
        tmp_path,
        "objective_auxiliary",
        quota,
        checkpoint_tokens=1,
        verification_required=True,
    )
    writer.add(encoded[0], {"record": 1})
    writer.close()

    resumed = _LaneWriter(
        tmp_path,
        "objective_auxiliary",
        quota,
        checkpoint_tokens=1,
        verification_required=True,
    )
    assert resumed.records == 1
    assert resumed.tokens == encoded[0].token_count
    resumed.add(encoded[1], {"record": 2})
    resumed.mark_complete()
    resumed.close()

    state = materialization_status(
        tmp_path,
        recipe=production_module.ProductionRecipe.for_testing(
            {lane: quota for lane in FROZEN_LANES},
            update_tokens=1,
            required_source_locks={lane: (f"lock_{lane}",) for lane in FROZEN_LANES},
        ),
    )
    assert state["lanes"]["objective_auxiliary"]["complete"] is True

    duplicate_root = tmp_path / "duplicate"
    duplicate = _LaneWriter(
        duplicate_root,
        "objective_auxiliary",
        encoded[0].token_count * 2,
        checkpoint_tokens=1 << 20,
        verification_required=True,
    )
    duplicate.add(encoded[0], {"record": 1})
    with pytest.raises(V2MaterializationError) as captured:
        duplicate.add(encoded[0], {"record": 2})
    duplicate.close()
    assert captured.value.code == "reasoning_cycle_fill"


def test_lane_compiler_uses_fresh_subset_to_meet_exact_quota(tmp_path):
    class _ListSource(_RecordSource):
        def __init__(self, records):
            self.records = records
            self.closed = False

        def next(self, cursor):
            index = int(cursor.get("index", 0))
            if index >= len(self.records):
                raise StopIteration
            return self.records[index], {"index": index + 1}

        def close(self):
            self.closed = True

    records = tuple(
        LaneRecord(
            f"text-{index}",
            "text",
            (LaneSegment(" a" * words),),
        )
        for index, words in enumerate((2, 4, 8))
    )
    lengths = tuple(_encode_record(get_tok(), record).token_count for record in records)
    assert lengths == (3, 5, 9)
    source = _ListSource(records)
    writer = _LaneWriter(
        tmp_path,
        "fineweb_edu",
        lengths[0] + lengths[2],
        checkpoint_tokens=1 << 20,
    )

    _compile_exact_lane(
        writer,
        source,
        finish_window=writer.quota,
        max_finish_candidates=3,
    )
    writer.close()

    checkpoint = json.loads(
        (tmp_path / "lanes" / "fineweb_edu" / "checkpoint.json").read_bytes()
    )
    assert checkpoint["complete"] is True
    assert checkpoint["tokens"] == writer.quota
    assert checkpoint["records"] == 2
    assert checkpoint["cursor"] == {"index": 3}
    assert source.closed is True


def test_exact_finish_interruption_replays_from_last_durable_checkpoint(tmp_path):
    class _ListSource(_RecordSource):
        def __init__(self, records):
            self.records = records

        def next(self, cursor):
            index = int(cursor.get("index", 0))
            if index >= len(self.records):
                raise StopIteration
            return self.records[index], {"index": index + 1}

    records = tuple(
        LaneRecord(
            f"text-{index}",
            "text",
            (LaneSegment(" a" * words),),
        )
        for index, words in enumerate((2, 4, 8))
    )
    lengths = tuple(_encode_record(get_tok(), record).token_count for record in records)
    writer = _LaneWriter(
        tmp_path,
        "fineweb_edu",
        lengths[0] + lengths[2],
        checkpoint_tokens=1 << 20,
    )
    original_add = writer.add
    installed = 0

    def interrupt_after_first_install(encoded, next_cursor):
        nonlocal installed
        original_add(encoded, next_cursor)
        installed += 1
        if installed == 1:
            raise RuntimeError("simulated exact-finish interruption")

    writer.add = interrupt_after_first_install
    with pytest.raises(RuntimeError, match="simulated exact-finish"):
        _compile_exact_lane(
            writer,
            _ListSource(records),
            finish_window=writer.quota,
            max_finish_candidates=3,
        )
    writer.close(checkpoint=False)

    checkpoint = json.loads(
        (tmp_path / "lanes" / "fineweb_edu" / "checkpoint.json").read_bytes()
    )
    assert checkpoint["tokens"] == 0
    resumed = _LaneWriter(
        tmp_path,
        "fineweb_edu",
        lengths[0] + lengths[2],
        checkpoint_tokens=1 << 20,
    )
    assert resumed.tokens == 0
    _compile_exact_lane(
        resumed,
        _ListSource(records),
        finish_window=resumed.quota,
        max_finish_candidates=3,
    )
    resumed.close()
    assert (
        json.loads(
            (tmp_path / "lanes" / "fineweb_edu" / "checkpoint.json").read_bytes()
        )["complete"]
        is True
    )


def test_lane_writer_rejects_solver_premises_detached_from_payloads(tmp_path):
    premises = (
        GraphTraversalPremise("fact:a", 0, "Q1", "P1", "Q2"),
        GraphTraversalPremise("fact:b", 1, "Q2", "P2", "Q3"),
    )
    record = LaneRecord(
        "detached-proof",
        "wikidata_path_reasoning_generator",
        (
            LaneSegment("first", "fact:a"),
            LaneSegment(" detached", "fact:other"),
        ),
        _solver_bundle(premises),
    )
    encoded = _encode_record(get_tok(), record)
    writer = _LaneWriter(
        tmp_path,
        "wikidata_path_reasoning",
        encoded.token_count,
        checkpoint_tokens=1 << 20,
        verification_required=True,
    )

    with pytest.raises(V2MaterializationError) as captured:
        writer.add(encoded, {"record": 1})
    writer.close(checkpoint=False)

    assert captured.value.code == "solver_premise_occurrence_mismatch"


def test_lane_compiler_fails_closed_when_fresh_records_exhaust(tmp_path):
    record = LaneRecord(
        "only-record",
        "text",
        (LaneSegment("one finite record"),),
    )

    class _FiniteSource(_RecordSource):
        def next(self, cursor):
            if cursor:
                raise StopIteration
            return record, {"done": True}

    token_count = _encode_record(get_tok(), record).token_count
    writer = _LaneWriter(
        tmp_path,
        "verified_synthetic_multihop",
        token_count + 1,
        checkpoint_tokens=1 << 20,
    )
    with pytest.raises(V2MaterializationError) as captured:
        _compile_exact_lane(
            writer,
            _FiniteSource(),
            finish_window=writer.quota,
            max_finish_candidates=2,
        )
    writer.close()
    assert captured.value.code == "fresh_source_exhausted"
    assert writer.tokens == 0


def test_wikidata_path_source_uses_functional_edges_and_solver_rows():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE functional_triples (
            ordinal INTEGER PRIMARY KEY,
            subject INTEGER NOT NULL,
            relation INTEGER NOT NULL,
            target INTEGER NOT NULL
        )
        """
    )
    connection.executemany(
        "INSERT INTO functional_triples VALUES (?, ?, ?, ?)",
        (
            (3, 1, 10, 2),
            (8, 2, 20, 3),
            (9, 2, 21, 4),
        ),
    )

    class _Index:
        pass

    index = _Index()
    index.connection = connection
    source = _WikidataPathSource(index)

    first, cursor = source.next({})
    second, _ = source.next(cursor)

    assert first.record_id == "wikidata-path:3:8"
    assert second.record_id == "wikidata-path:3:9"
    assert first.verification["family"] == "graph_path_traversal"
    assert first.verification["proof"]["conclusion"] == {"endpoint": "Q3"}
    assert [row["fact_id"] for row in first.verification["premises"]] == [
        "wikidata:Q1:P10:Q2",
        "wikidata:Q2:P20:Q3",
    ]
    connection.close()


def test_wikidata_index_materializes_selected_training_triples_once(tmp_path):
    stage = tmp_path / "stage"
    files = stage / "wikidata5m" / "files"
    selection = stage / "wikidata5m" / "selection"
    files.mkdir(parents=True)
    selection.mkdir()
    (files / "train.txt").write_text(
        "Q1\tP10\tQ2\nQ2\tP20\tQ3\nQ9\tP90\tQ10\n",
        encoding="utf-8",
    )
    (selection / "train.keep.u8").write_bytes(b"\x01\x01\x00")
    (stage / "wikidata5m" / "selection-manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "input_path": "train.txt",
                        "rows": 3,
                        "sidecar": {"path": "selection/train.keep.u8"},
                    }
                ],
                "format": "memorysplit-v2-wikidata-complete-once-selection",
                "totals": {"distinct_training_triples": 2},
            }
        ),
        encoding="utf-8",
    )

    index = _WikidataIndex(stage, tmp_path / "work", "fixture-lock")
    index.ensure()
    assert index.triple_count == 2
    assert index.connection.execute(
        "SELECT subject, relation, target FROM triples ORDER BY ordinal"
    ).fetchall() == [(1, 10, 2), (2, 20, 3)]
    index.close()

    resumed = _WikidataIndex(stage, tmp_path / "work", "fixture-lock")
    resumed.ensure()
    assert resumed.triple_count == 2
    resumed.close()


def test_wikidata_graph_next_group_uses_ordered_index_seeks():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE triples (
            ordinal INTEGER PRIMARY KEY,
            subject INTEGER NOT NULL,
            relation INTEGER NOT NULL,
            target INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE UNIQUE INDEX triples_address_target "
        "ON triples(subject, relation, target)"
    )
    connection.executemany(
        "INSERT INTO triples VALUES (?, ?, ?, ?)",
        (
            (0, 1, 10, 2),
            (1, 1, 10, 3),
            (2, 1, 12, 4),
            (3, 3, 2, 5),
        ),
    )

    class _Index:
        pass

    index = _Index()
    index.connection = connection
    source = _WikidataGraphSource(index)

    assert source._next_group(-1, -1) == (1, 10)
    assert source._next_group(1, 10) == (1, 12)
    assert source._next_group(1, 12) == (3, 2)
    assert source._next_group(3, 2) is None

    plans = (
        connection.execute(
            f"EXPLAIN QUERY PLAN {source._NEXT_RELATION_SQL}",
            (1, 10),
        ).fetchall(),
        connection.execute(
            f"EXPLAIN QUERY PLAN {source._NEXT_SUBJECT_SQL}",
            (1,),
        ).fetchall(),
    )
    for rows in plans:
        detail = " ".join(str(row[3]).upper() for row in rows)
        assert "SEARCH" in detail
        assert "SCAN" not in detail
    connection.close()


def test_wikidata_graph_pages_each_selected_triple_once_and_resumes(
    monkeypatch,
):
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE triples (
            ordinal INTEGER PRIMARY KEY,
            subject INTEGER NOT NULL,
            relation INTEGER NOT NULL,
            target INTEGER NOT NULL
        )
        """
    )
    triples = (
        (0, 1, 10, 2),
        (1, 1, 10, 3),
        (2, 1, 10, 4),
        (3, 2, 20, 5),
    )
    connection.executemany("INSERT INTO triples VALUES (?, ?, ?, ?)", triples)

    class _Index:
        pass

    index = _Index()
    index.connection = connection
    source = _WikidataGraphSource(index)
    tok = get_tok()
    single_record_sizes = [
        _encode_record(
            tok,
            source._record(subject, relation, page, ((ordinal, target),)),
        ).token_count
        for page, (ordinal, subject, relation, target) in enumerate(triples[:3])
    ]
    pair_size = _encode_record(
        tok,
        source._record(1, 10, 0, ((0, 2), (1, 3))),
    ).token_count
    page_limit = max(single_record_sizes)
    assert pair_size > page_limit
    monkeypatch.setattr(materialize_module, "MAX_RECORD_TOKENS", page_limit)

    first, cursor = source.next({})
    resumed = _WikidataGraphSource(index)
    records = [first]
    while True:
        try:
            record, cursor = resumed.next(cursor)
        except StopIteration:
            break
        records.append(record)

    encoded = [_encode_record(tok, record) for record in records]
    assert all(record.token_count <= page_limit for record in encoded)
    assert [
        parse_serialized_action(tok.encode(record.record.segments[1].text), tok).page
        for record in encoded
    ] == [0, 1, 2, 0]
    assert [
        occurrence.fact_id for record in encoded for occurrence in record.facts
    ] == [
        "wikidata:Q1:P10:Q2",
        "wikidata:Q1:P10:Q3",
        "wikidata:Q1:P10:Q4",
        "wikidata:Q2:P20:Q5",
    ]
    for record, (ordinal, subject, relation, target) in zip(encoded, triples):
        row, fact_id = materialize_module._wikidata_row(
            subject,
            relation,
            target,
            ordinal,
        )
        scalar_surface = next(
            segment.text
            for segment in materialize_module._graph_record_segments(
                tok,
                row,
                fact_id,
            )
            if segment.fact_id == fact_id
        )
        paged_surface = next(
            segment.text
            for segment in record.record.segments
            if segment.fact_id == fact_id
        )
        assert paged_surface == scalar_surface
    connection.close()


def test_repository_synthetic_generators_emit_fresh_encodable_records():
    tok = get_tok()
    sources = (
        _SyntheticGraphSource(),
        _SyntheticMultihopSource(),
        _RelationalRefinementSource(),
    )

    for source in sources:
        first, cursor = source.next({})
        second, _ = source.next(cursor)
        assert first.record_id != second.record_id
        assert _encode_record(tok, first).token_count > 0
        assert _encode_record(tok, second).token_count > 0
    assert sources[1].next({})[0].verification["kind"] == "solver"
    refinement = sources[2].next({})[0]
    assert refinement.verification["kind"] == "solver"
    # Balanced equality reads two independent entity slots.  Rendering both
    # reads from slot zero would make the token trace disagree with the proof.
    first_action = parse_serialized_action(tok.encode(refinement.segments[1].text), tok)
    second_action = parse_serialized_action(
        tok.encode(refinement.segments[6].text), tok
    )
    assert (first_action.source_slot, second_action.source_slot) == (0, 1)
    actions = [
        parse_serialized_action(tok.encode(refinement.segments[index].text), tok)
        for index in (1, 6, 11, 16, 21, 26)
    ]
    assert [action.read for action in actions] == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert [action.halt for action in actions] == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]


def test_generated_training_world_ranges_exclude_protected_evaluation():
    from corpusgen.relational_build import _EVAL_WORLD_ID

    assert _EVAL_WORLD_ID < _GRAPH_WORLD_OFFSET
    assert _GRAPH_WORLD_OFFSET < _MULTIHOP_WORLD_OFFSET
    assert _MULTIHOP_WORLD_OFFSET < _REFINEMENT_WORLD_OFFSET


def test_generated_generator_and_solver_locks_bind_real_repository_bytes():
    payloads = _generated_lock_payloads(
        {"provider": {"python": "test"}},
        {"python": "test", "pyarrow": "test", "tiktoken": "test"},
    )

    assert set(payloads) == {
        "reasoning_solver",
        "relational_refinement_generator",
        "synthetic_graph_generator",
        "verified_synthetic_multihop_generator",
        "wikidata_path_reasoning_generator",
    }
    for source_id, payload in payloads.items():
        production_module._validate_source_lock(source_id, payload)
        lock = json.loads(payload)
        assert lock["artifacts"]
        assert all(item["bytes"] > 0 for item in lock["artifacts"])
        assert lock["policy"]["materializer_version"]
        if "://" in lock["repository"]:
            authority = lock["repository"].split("://", 1)[1].split("/", 1)[0]
            assert "@" not in authority
    assert json.loads(payloads["reasoning_solver"])["kind"] == "solver"
    graph_policy = json.loads(payloads["synthetic_graph_generator"])["policy"]
    assert graph_policy["candidate_fact_reuse"] is False
    assert graph_policy["world_schedule"] == (
        "emit_each_fact_candidate_once_then_advance"
    )
    multihop_policy = json.loads(payloads["verified_synthetic_multihop_generator"])[
        "policy"
    ]
    assert multihop_policy["path_edges_unique_within_record"] is True
    assert multihop_policy["cross_record_edge_reuse"] == (
        "allowed_within_each_fresh_world"
    )
    solver_policy = json.loads(payloads["reasoning_solver"])["policy"]
    assert solver_policy["objective_determinism_adapters"] == {
        "deepmind_mathematics_generator": (
            "ordinal_seed_python_numpy_sympy;"
            "stable_entity_insertion_order_before_native_seeded_shuffle;"
            "integral_float_randint_compatibility"
        )
    }


def test_smoke_materializes_deterministic_45_artifact_contract(tmp_path):
    first_source = tmp_path / "first-source"
    second_source = tmp_path / "second-source"

    first = materialize_v2_smoke(first_source, tmp_path / "first-work")
    second = materialize_v2_smoke(second_source, tmp_path / "second-work")

    assert first["scientific_result"] is False
    assert first["production_preflight"]["ready"] is True
    assert second["production_preflight"]["ready"] is True
    first_files = _source_files(first_source)
    second_files = _source_files(second_source)
    assert first_files == second_files
    assert len(first_files) == 46
    assert "source-manifest.json" in first_files
    assert len(first_files) - 1 == 45
    assert {path for path in first_files if path.startswith("materialized/")} == {
        f"materialized/{lane}.{suffix}"
        for lane in FROZEN_LANES
        for suffix in ("tokens.bin", "split90.weights.bin")
    }

    dose = first["production_preflight"]["route_dose"]
    assert Fraction(
        dose["distinct_external_facts"],
        dose["distinct_facts"],
    ) >= Fraction(9, 10)
    assert Fraction(
        dose["external_burden"]["numerator"],
        dose["external_burden"]["denominator"],
    ) / Fraction(
        dose["total_burden"]["numerator"],
        dose["total_burden"]["denominator"],
    ) >= Fraction(9, 10)

    shared_rows = {}
    for lane in ("wikidata_graph", "wikidata_path_reasoning"):
        rows = [
            json.loads(line)
            for line in (first_source / "ledgers" / f"{lane}.routes.jsonl")
            .read_bytes()
            .splitlines()
        ]
        shared_rows[lane] = {
            row["fact_id"]: row
            for row in rows
            if row["fact_id"] in {"smoke:shared:a", "smoke:shared:b"}
        }
    assert shared_rows["wikidata_graph"] == shared_rows["wikidata_path_reasoning"]


def test_publication_resumes_an_interrupted_cross_filesystem_copy(
    tmp_path,
    monkeypatch,
):
    reference = tmp_path / "reference"
    work = tmp_path / "work"
    materialize_v2_smoke(reference, work)
    quotas = {
        lane: json.loads((work / "lanes" / lane / "checkpoint.json").read_bytes())[
            "quota"
        ]
        for lane in FROZEN_LANES
    }
    lock_mapping = {
        "fineweb_edu": ("smoke_fineweb",),
        "finemath": ("smoke_finemath",),
        "wikidata_graph": ("smoke_wikidata",),
        "synthetic_graph": ("smoke_synthetic",),
        "verified_synthetic_multihop": ("smoke_verified", "smoke_solver"),
        "wikidata_path_reasoning": (
            "smoke_wikidata",
            "smoke_path",
            "smoke_solver",
        ),
        "relational_refinement": ("smoke_refinement", "smoke_solver"),
        "objective_auxiliary": ("smoke_objective",),
    }
    recipe = production_module.ProductionRecipe.for_testing(
        quotas,
        update_tokens=1,
        required_source_locks=lock_mapping,
    )
    lock_ids = {
        source_id for source_ids in lock_mapping.values() for source_id in source_ids
    }
    lock_payloads = {
        source_id: materialize_module._smoke_lock(source_id) for source_id in lock_ids
    }
    destination = tmp_path / "resumed"
    original_copy = materialize_module._copy_or_link
    copied = 0

    def interrupt_after_copy(source, target):
        nonlocal copied
        original_copy(source, target)
        copied += 1
        if copied == 1:
            raise RuntimeError("simulated publication interruption")

    with monkeypatch.context() as patch:
        patch.setattr(materialize_module, "_copy_or_link", interrupt_after_copy)
        with pytest.raises(RuntimeError, match="publication interruption"):
            materialize_module._publish_source_root(
                work,
                destination,
                recipe,
                lock_payloads,
                seal=True,
            )

    report = materialize_module._publish_source_root(
        work,
        destination,
        recipe,
        lock_payloads,
        seal=True,
    )
    assert report["ready"] is True
    assert _source_files(destination) == _source_files(reference)


def test_routing_cache_rebuilds_a_same_size_corrupt_sidecar(tmp_path):
    source = tmp_path / "source"
    work = tmp_path / "work"
    result = materialize_v2_smoke(source, work)
    quotas = {
        lane: json.loads((work / "lanes" / lane / "checkpoint.json").read_bytes())[
            "quota"
        ]
        for lane in FROZEN_LANES
    }
    recipe = production_module.ProductionRecipe.for_testing(
        quotas,
        update_tokens=1,
        required_source_locks={
            "fineweb_edu": ("smoke_fineweb",),
            "finemath": ("smoke_finemath",),
            "wikidata_graph": ("smoke_wikidata",),
            "synthetic_graph": ("smoke_synthetic",),
            "verified_synthetic_multihop": (
                "smoke_verified",
                "smoke_solver",
            ),
            "wikidata_path_reasoning": (
                "smoke_wikidata",
                "smoke_path",
                "smoke_solver",
            ),
            "relational_refinement": (
                "smoke_refinement",
                "smoke_solver",
            ),
            "objective_auxiliary": ("smoke_objective",),
        },
    )
    split_path = work / "lanes" / "fineweb_edu" / "split90.weights.bin"
    original = split_path.read_bytes()
    split_path.write_bytes(bytes((original[0] ^ 1,)) + original[1:])

    rebuilt = materialize_module._route_all_lanes(work, recipe)

    assert split_path.read_bytes() == original
    assert rebuilt["output_identity"] == result["routing"]["output_identity"]


def test_materializer_cli_runs_smoke_and_reports_status(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "materialize_memorysplit_v2.py"
    source = tmp_path / "source"
    work = tmp_path / "work"

    smoke = subprocess.run(
        [
            sys.executable,
            str(script),
            "smoke",
            "--source-root",
            str(source),
            "--work-root",
            str(work),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert smoke.returncode == 0, smoke.stderr
    assert json.loads(smoke.stdout)["production_preflight"]["ready"] is True
    status = subprocess.run(
        [
            sys.executable,
            str(script),
            "status",
            "--work-root",
            str(work),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert status.returncode == 0, status.stderr
    value = json.loads(status.stdout)
    assert all(value["lanes"][lane]["complete"] for lane in FROZEN_LANES)
    assert value["routing_complete"] is True


def test_preflight_rejects_inconsistent_cross_lane_route_evidence(tmp_path):
    source = tmp_path / "source"
    materialize_v2_smoke(source, tmp_path / "work")
    route_path = source / "ledgers" / "wikidata_path_reasoning.routes.jsonl"
    rows = [json.loads(line) for line in route_path.read_bytes().splitlines()]
    for row in rows:
        if row["fact_id"] == "smoke:shared:a":
            row["burden_bits"]["numerator"] += 1
    route_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    (source / "source-manifest.json").unlink()

    recipe = production_module.ProductionRecipe.for_testing(
        {
            lane: (source / "materialized" / f"{lane}.tokens.bin").stat().st_size // 2
            for lane in FROZEN_LANES
        },
        update_tokens=1,
        required_source_locks={
            "fineweb_edu": ("smoke_fineweb",),
            "finemath": ("smoke_finemath",),
            "wikidata_graph": ("smoke_wikidata",),
            "synthetic_graph": ("smoke_synthetic",),
            "verified_synthetic_multihop": (
                "smoke_verified",
                "smoke_solver",
            ),
            "wikidata_path_reasoning": (
                "smoke_wikidata",
                "smoke_path",
                "smoke_solver",
            ),
            "relational_refinement": (
                "smoke_refinement",
                "smoke_solver",
            ),
            "objective_auxiliary": ("smoke_objective",),
        },
    )
    with pytest.raises(production_module.ProductionPreflightError) as captured:
        production_module.seal_production_sources(source, recipe=recipe)
    assert "inconsistent global routing evidence" in str(captured.value)
