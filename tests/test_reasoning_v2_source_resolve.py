"""Offline metadata pinning of the reasoning-v2 source lock.

Every test in this file runs without network access. The recorded metadata
fixture mirrors the response shapes of ``huggingface.co`` and
``git ls-remote``; the FineWeb-Edu and Wikidata5M digests inside it are the
independently reviewed values already frozen in ``source_lock.py``, so the
tests can prove that LFS metadata reproduces a human-reviewed pin.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from corpusgen.parallel.canonical import canonical_json_bytes
from corpusgen.reasoning_v2 import source_lock as source_lock_module
from corpusgen.reasoning_v2.source_lock import (
    FIXED_FINEWEB_FILES,
    FIXED_WIKIDATA_FILES,
    reviewed_source_catalog_sha256,
    verify_source_tree,
)
from corpusgen.reasoning_v2.source_resolve import (
    METADATA_RECORD_FORMAT,
    PIN_PLAN_FORMAT,
    HttpsSourceMetadataClient,
    RecordedMetadataClient,
    SourcePinPlan,
    UnpinnedSourceError,
    git_ls_remote_argv,
    huggingface_revision_url,
    huggingface_tree_url,
    load_metadata_record,
    metadata_egress_hosts,
    resolve_source_pins,
    source_lock_entries,
)
from reasoning_v2_fixtures import (  # noqa: F401
    FixtureSourceLock,
    fake_public_resolver,
    fixed_contract_environment,
    fixture_source_lock,
    full_recipe,
)
from scripts.resolve_source_lock import main


ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "tests/fixtures/source_metadata/reasoning-v2-source-metadata.json"

FINEWEB_BYTES = 6_456_837_861
FINEMATH_CANDIDATE_BYTES = 8_000_000_013
GIT_SOURCES = (
    "arc_agi_1",
    "arc_agi_2",
    "clrs_text",
    "conceptarc",
    "deepmind_mathematics_generator",
    "prontoqa",
    "reasoning_gym_exact_answer",
    "ruletaker",
)


@pytest.fixture
def record() -> dict:
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def client(record) -> RecordedMetadataClient:
    return RecordedMetadataClient(record)


@pytest.fixture
def plan(client) -> SourcePinPlan:
    return resolve_source_pins(client)


def _write_record(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "metadata.json"
    path.write_bytes(json.dumps(value, indent=2).encode("utf-8"))
    return path


def _source(plan: SourcePinPlan, source_id: str):
    return next(row for row in plan.sources if row.source_id == source_id)


def _bindings(plan: SourcePinPlan, source_id: str) -> tuple:
    return tuple(row for row in plan.unpinnable if row.source_id == source_id)


class _RefusingClient:
    """Any metadata call at all is a contract violation for this client."""

    def git_ls_remote(self, repository: str, ref: str) -> str:
        raise AssertionError(f"unexpected git metadata call: {repository} {ref}")

    def huggingface_dataset_revision(self, repository: str, revision: str) -> object:
        raise AssertionError(f"unexpected HF revision call: {repository}@{revision}")

    def huggingface_dataset_tree(self, repository: str, revision: str) -> object:
        raise AssertionError(f"unexpected HF tree call: {repository}@{revision}")


class _CountingClient:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[tuple[str, str, str]] = []

    def git_ls_remote(self, repository: str, ref: str) -> str:
        self.calls.append(("git_ls_remote", repository, ref))
        return self.inner.git_ls_remote(repository, ref)

    def huggingface_dataset_revision(self, repository: str, revision: str) -> object:
        self.calls.append(("huggingface_dataset_revision", repository, revision))
        return self.inner.huggingface_dataset_revision(repository, revision)

    def huggingface_dataset_tree(self, repository: str, revision: str) -> object:
        self.calls.append(("huggingface_dataset_tree", repository, revision))
        return self.inner.huggingface_dataset_tree(repository, revision)


def test_plan_covers_every_reviewed_source_and_commits_to_the_catalog(plan):
    assert {row.source_id for row in plan.sources} == set(
        source_lock_module._EXPECTED_SOURCE_IDS
    )
    assert plan.format == PIN_PLAN_FORMAT
    assert plan.schema_version == 1
    assert plan.dataset_id == source_lock_module.DATASET_ID
    assert plan.source_catalog_sha256 == reviewed_source_catalog_sha256()


def test_wikidata_is_fully_pinnable_without_downloading_any_content(plan):
    entry = _source(plan, "wikidata5m")
    assert _bindings(plan, "wikidata5m") == ()
    assert entry.inventory_complete is True
    assert entry.revision == source_lock_module._FIXED_IDENTITIES["wikidata5m"][2]
    assert entry.revision_provenance == "reviewed_catalog"
    by_path = {row.path: row for row in entry.files}
    assert set(by_path) == {
        "Wikidata-CC0-1.0.txt",
        "wikidata5m_alias.tar.gz",
        "wikidata5m_inductive.tar.gz",
        "wikidata5m_transductive.tar.gz",
    }
    for path, expected in FIXED_WIKIDATA_FILES.items():
        assert by_path[path].sha256 == expected["sha256"]
        assert by_path[path].bytes == expected["bytes"]
        assert by_path[path].provenance == "huggingface_lfs"
    notice = source_lock_module.WIKIDATA_NOTICE_PATH.read_bytes()
    assert by_path["Wikidata-CC0-1.0.txt"].sha256 == hashlib.sha256(notice).hexdigest()
    assert by_path["Wikidata-CC0-1.0.txt"].provenance == "local_contract"


def test_lfs_metadata_reproduces_the_reviewed_fineweb_digests(plan):
    entry = _source(plan, "fineweb_edu")
    by_path = {row.path: row for row in entry.files}
    assert set(by_path) == set(FIXED_FINEWEB_FILES)
    for path, expected in FIXED_FINEWEB_FILES.items():
        assert by_path[path].sha256 == expected["sha256"]
        assert by_path[path].bytes == expected["bytes"]
    assert sum(row.bytes for row in entry.files) == FINEWEB_BYTES


def test_unrequested_repository_files_are_never_pinned(plan):
    entry = _source(plan, "fineweb_edu")
    assert all(
        not row.path.startswith("data/")
        for row in (*entry.files, *entry.candidate_files)
    )


def test_non_lfs_license_file_is_refused_with_its_exact_byte_cost(plan):
    bindings = _bindings(plan, "fineweb_edu")
    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.binding == "file_digest"
    assert binding.path == "README.md"
    assert binding.bytes_required == 19_811
    assert binding.byte_estimate == "exact"
    assert "sha-256" in binding.reason.lower()
    assert _source(plan, "fineweb_edu").inventory_complete is False


def test_metadata_contradicting_a_reviewed_pin_fails_loudly(record):
    tree = record["huggingface_tree"]["HuggingFaceFW/fineweb-edu"][
        "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    ]
    row = next(
        entry
        for entry in tree
        if entry["path"] == "sample/10BT/000_00000.parquet"
    )
    row["lfs"]["oid"] = "f" * 64
    with pytest.raises(ValueError, match="reviewed file digest drift"):
        resolve_source_pins(RecordedMetadataClient(record))


def test_metadata_contradicting_a_reviewed_byte_count_fails_loudly(record):
    tree = record["huggingface_tree"]["intfloat/wikidata5m"][
        "6b2b09672129e280c0c9da97ab58154e9d535e6b"
    ]
    row = next(
        entry for entry in tree if entry["path"] == "wikidata5m_alias.tar.gz"
    )
    row["lfs"]["size"] = row["size"] = 1
    with pytest.raises(ValueError, match="reviewed file digest drift"):
        resolve_source_pins(RecordedMetadataClient(record))


def test_lfs_pointer_size_disagreement_is_refused(record):
    tree = record["huggingface_tree"]["HuggingFaceTB/finemath"][
        "8924e1b2f0f5c7bdd6c6ff9b8b0a2a5f3d1c0e7a"
    ]
    row = next(
        entry
        for entry in tree
        if entry["path"] == "finemath-4plus/train-00000-of-00004.parquet"
    )
    row["lfs"]["size"] = row["size"] + 1
    with pytest.raises(ValueError, match="byte count disagrees"):
        resolve_source_pins(RecordedMetadataClient(record))


def test_missing_required_file_in_the_tree_is_refused(record):
    key = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    tree = record["huggingface_tree"]["HuggingFaceFW/fineweb-edu"][key]
    record["huggingface_tree"]["HuggingFaceFW/fineweb-edu"][key] = [
        entry
        for entry in tree
        if entry["path"] != "sample/10BT/001_00000.parquet"
    ]
    with pytest.raises(ValueError, match="required file is absent"):
        resolve_source_pins(RecordedMetadataClient(record))


def test_git_sources_pin_the_revision_but_never_the_file_inventory(plan):
    for source_id in GIT_SOURCES:
        entry = _source(plan, source_id)
        assert entry.files == ()
        assert entry.candidate_files == ()
        assert entry.inventory_complete is False
        kinds = {row.binding for row in _bindings(plan, source_id)}
        assert "file_inventory" in kinds
        reason = next(
            row.reason
            for row in _bindings(plan, source_id)
            if row.binding == "file_inventory"
        )
        assert "sha-1" in reason.lower()


def test_reviewed_fixed_revisions_need_no_git_metadata_call(record):
    counting = _CountingClient(RecordedMetadataClient(record))
    resolve_source_pins(counting)
    contacted = {repository for _kind, repository, _ref in counting.calls}
    for source_id in ("arc_agi_1", "arc_agi_2", "conceptarc"):
        repository = source_lock_module._FIXED_IDENTITIES[source_id][1]
        assert repository not in contacted


def test_unfixed_git_revisions_come_from_ls_remote(plan):
    entry = _source(plan, "clrs_text")
    assert entry.revision == "1a2b3c4d5e6f70819202a3b4c5d6e7f809111213"
    assert entry.revision_provenance == "git_ls_remote"


def test_ambiguous_ls_remote_output_is_refused(record):
    record["git_ls_remote"]["https://github.com/asaparov/prontoqa.git"]["HEAD"] = (
        "3c4d5e6f708192a3b4c5d6e7f809111213141516\tHEAD\n"
        "3c4d5e6f708192a3b4c5d6e7f809111213141516\trefs/heads/main\n"
    )
    with pytest.raises(ValueError, match="one immutable commit"):
        resolve_source_pins(RecordedMetadataClient(record))


def test_symbolic_ls_remote_answer_is_refused(record):
    record["git_ls_remote"]["https://github.com/asaparov/prontoqa.git"]["HEAD"] = (
        "ref: refs/heads/main\tHEAD\n"
    )
    with pytest.raises(ValueError, match="one immutable commit"):
        resolve_source_pins(RecordedMetadataClient(record))


def test_ruletaker_reports_its_separate_https_artifact_host(plan):
    bindings = {row.binding: row for row in _bindings(plan, "ruletaker")}
    assert set(bindings) == {"file_inventory", "file_digest"}
    archive = bindings["file_digest"]
    assert archive.path == source_lock_module._RULETAKER_ARCHIVE
    assert archive.byte_estimate == "unknown"
    assert archive.bytes_required is None
    assert "aristo-data-public.s3-us-west-2.amazonaws.com" in archive.reason


def test_finemath_revision_and_every_candidate_shard_pin_from_metadata(plan):
    entry = _source(plan, "finemath")
    assert entry.revision == "8924e1b2f0f5c7bdd6c6ff9b8b0a2a5f3d1c0e7a"
    assert entry.revision_provenance == "huggingface_revision"
    assert entry.files == ()
    assert tuple(row.path for row in entry.candidate_files) == (
        "finemath-3plus/train-00000-of-00002.parquet",
        "finemath-3plus/train-00001-of-00002.parquet",
        "finemath-4plus/train-00000-of-00004.parquet",
        "finemath-4plus/train-00001-of-00004.parquet",
        "finemath-4plus/train-00002-of-00004.parquet",
        "finemath-4plus/train-00003-of-00004.parquet",
    )
    assert all(row.provenance == "huggingface_lfs" for row in entry.candidate_files)
    assert sum(row.bytes for row in entry.candidate_files) == FINEMATH_CANDIDATE_BYTES


def test_finemath_selection_proof_is_refused_with_a_byte_upper_bound(plan):
    bindings = {row.binding: row for row in _bindings(plan, "finemath")}
    assert set(bindings) == {"file_digest", "finemath_selection"}
    selection = bindings["finemath_selection"]
    assert selection.byte_estimate == "upper_bound"
    assert selection.bytes_required == FINEWEB_BYTES + FINEMATH_CANDIDATE_BYTES
    assert "cross-dedup" in selection.reason.lower()
    assert _source(plan, "finemath").inventory_complete is False


def test_finemath_card_without_odc_by_is_refused(record):
    record["huggingface_revision"]["HuggingFaceTB/finemath"]["main"]["cardData"][
        "license"
    ] = "cc-by-4.0"
    with pytest.raises(ValueError, match="ODC-By"):
        resolve_source_pins(RecordedMetadataClient(record))


def test_finemath_shard_sequence_gap_is_refused(record):
    key = "8924e1b2f0f5c7bdd6c6ff9b8b0a2a5f3d1c0e7a"
    tree = record["huggingface_tree"]["HuggingFaceTB/finemath"][key]
    record["huggingface_tree"]["HuggingFaceTB/finemath"][key] = [
        entry
        for entry in tree
        if entry["path"] != "finemath-4plus/train-00002-of-00004.parquet"
    ]
    with pytest.raises(ValueError, match="shard sequence"):
        resolve_source_pins(RecordedMetadataClient(record))


def test_a_file_without_lfs_metadata_never_becomes_a_pin(record):
    key = "6b2b09672129e280c0c9da97ab58154e9d535e6b"
    tree = record["huggingface_tree"]["intfloat/wikidata5m"][key]
    row = next(
        entry for entry in tree if entry["path"] == "wikidata5m_alias.tar.gz"
    )
    row.pop("lfs")
    with pytest.raises(ValueError, match="reviewed file digest drift"):
        resolve_source_pins(RecordedMetadataClient(record))


def test_plan_is_incomplete_and_refuses_to_emit_source_entries(plan):
    assert plan.complete is False
    with pytest.raises(UnpinnedSourceError, match="finemath"):
        source_lock_entries(plan)


def test_plan_json_is_canonical_and_content_addressed(plan):
    payload = plan.to_bytes()
    assert payload == canonical_json_bytes(plan.as_dict())
    assert plan.sha256 == hashlib.sha256(payload).hexdigest()
    assert SourcePinPlan.from_dict(json.loads(payload)).to_bytes() == payload


def test_plan_orders_sources_and_bindings_bytewise(plan):
    ordered = tuple(row.source_id for row in plan.sources)
    assert ordered == tuple(sorted(ordered, key=str.encode))
    keys = tuple((row.source_id, row.binding, row.path or "") for row in plan.unpinnable)
    assert keys == tuple(sorted(keys))


def test_byte_budget_separates_freeze_cost_from_build_cost(plan):
    budget = plan.byte_budget()
    assert budget["pinned_bytes"] == (
        FINEWEB_BYTES
        + sum(int(row["bytes"]) for row in FIXED_WIKIDATA_FILES.values())
        + len(source_lock_module.WIKIDATA_NOTICE_PATH.read_bytes())
        + FINEMATH_CANDIDATE_BYTES
    )
    assert budget["freeze_bytes_exact"] == 19_811 + 12_345
    assert budget["freeze_bytes_upper_bound"] == (
        budget["freeze_bytes_exact"] + FINEWEB_BYTES + FINEMATH_CANDIDATE_BYTES
    )
    assert budget["freeze_bindings_without_estimate"] == len(GIT_SOURCES) + 1


def test_evidence_digest_binds_the_plan_to_the_metadata_it_consumed(record):
    sink: list[dict] = []
    plan = resolve_source_pins(RecordedMetadataClient(record), evidence_sink=sink)
    assert len(sink) == 1
    assert sink[0]["format"] == METADATA_RECORD_FORMAT
    assert plan.evidence_sha256 == hashlib.sha256(
        canonical_json_bytes(sink[0])
    ).hexdigest()
    replayed = resolve_source_pins(RecordedMetadataClient(sink[0]))
    assert replayed.to_bytes() == plan.to_bytes()


def test_metadata_calls_are_the_documented_minimum(record):
    counting = _CountingClient(RecordedMetadataClient(record))
    resolve_source_pins(counting)
    assert counting.calls == [
        (
            "huggingface_dataset_tree",
            "HuggingFaceFW/fineweb-edu",
            "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        ),
        ("huggingface_dataset_revision", "HuggingFaceTB/finemath", "main"),
        (
            "huggingface_dataset_tree",
            "HuggingFaceTB/finemath",
            "8924e1b2f0f5c7bdd6c6ff9b8b0a2a5f3d1c0e7a",
        ),
        (
            "huggingface_dataset_tree",
            "intfloat/wikidata5m",
            "6b2b09672129e280c0c9da97ab58154e9d535e6b",
        ),
        ("git_ls_remote", "https://github.com/google-deepmind/clrs.git", "HEAD"),
        ("git_ls_remote", "https://github.com/allenai/ruletaker.git", "HEAD"),
        ("git_ls_remote", "https://github.com/asaparov/prontoqa.git", "HEAD"),
        (
            "git_ls_remote",
            "https://github.com/open-thought/reasoning-gym.git",
            "HEAD",
        ),
        (
            "git_ls_remote",
            "https://github.com/google-deepmind/mathematics_dataset.git",
            "HEAD",
        ),
    ]


def test_resolution_touches_nothing_but_the_injected_client(record, monkeypatch):
    def _forbidden(*_arguments, **_keywords):
        raise AssertionError("source pin resolution opened its own transport")

    monkeypatch.setattr("urllib.request.urlopen", _forbidden)
    monkeypatch.setattr("subprocess.run", _forbidden)
    monkeypatch.setattr("subprocess.check_output", _forbidden)
    resolve_source_pins(RecordedMetadataClient(record))


def test_recorded_client_refuses_to_invent_a_missing_response(record):
    del record["git_ls_remote"]["https://github.com/asaparov/prontoqa.git"]
    with pytest.raises(ValueError, match="metadata record has no git_ls_remote"):
        resolve_source_pins(RecordedMetadataClient(record))


def test_metadata_record_rejects_a_foreign_format(tmp_path, record):
    record["format"] = "something-else"
    with pytest.raises(ValueError, match="metadata record format"):
        load_metadata_record(_write_record(tmp_path, record))


def test_metadata_record_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_bytes(b'{"format": "a", "format": "b"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_metadata_record(path)


def test_metadata_egress_is_two_hosts():
    assert metadata_egress_hosts() == ("github.com", "huggingface.co")


def test_url_and_argv_builders_are_the_whole_egress_surface():
    assert huggingface_tree_url("HuggingFaceTB/finemath", "a" * 40) == (
        "https://huggingface.co/api/datasets/HuggingFaceTB/finemath/tree/"
        + "a" * 40
        + "?expand=1&recursive=1"
    )
    assert huggingface_revision_url("HuggingFaceTB/finemath", "main") == (
        "https://huggingface.co/api/datasets/HuggingFaceTB/finemath/revision/main"
    )
    assert git_ls_remote_argv("https://github.com/asaparov/prontoqa.git", "HEAD") == (
        "git",
        "ls-remote",
        "--exit-code",
        "https://github.com/asaparov/prontoqa.git",
        "HEAD",
    )


@pytest.mark.parametrize(
    "repository",
    ["../escape", "HuggingFaceTB/finemath?x=1", "HuggingFaceTB", "a/b/c", ""],
)
def test_url_builders_refuse_unsafe_repository_identifiers(repository):
    with pytest.raises(ValueError, match="dataset identifier"):
        huggingface_tree_url(repository, "a" * 40)


@pytest.mark.parametrize("revision", ["main~1", "a/b", "", "refs/heads/main"])
def test_url_builders_refuse_unsafe_revisions(revision):
    with pytest.raises(ValueError, match="revision"):
        huggingface_tree_url("HuggingFaceTB/finemath", revision)


def test_https_client_never_sends_ambient_credentials(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "must-not-be-consumed")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "must-not-be-consumed")
    seen: list[object] = []

    class _Response:
        def __init__(self, url: str) -> None:
            self._url = url

        def geturl(self) -> str:
            return self._url

        def read(self, _limit: int | None = None) -> bytes:
            return b'{"sha": "' + b"a" * 40 + b'"}'

        def __enter__(self):
            return self

        def __exit__(self, *_exception) -> bool:
            return False

    def _urlopen(request, timeout=None):
        seen.append(request)
        return _Response(request.full_url)

    monkeypatch.setattr(
        "corpusgen.reasoning_v2.source_resolve.urllib.request.urlopen",
        _urlopen,
    )
    client = HttpsSourceMetadataClient()
    assert client.huggingface_dataset_revision("HuggingFaceTB/finemath", "main") == {
        "sha": "a" * 40
    }
    request = seen[0]
    assert request.full_url == huggingface_revision_url(
        "HuggingFaceTB/finemath", "main"
    )
    assert not any(
        key.lower() in {"authorization", "cookie"} for key in request.headers
    )


def test_verification_catches_a_metadata_derived_hash_that_the_bytes_contradict(
    fixture_source_lock: FixtureSourceLock,
):
    lock = fixture_source_lock.lock
    root = fixture_source_lock.download_root
    assert verify_source_tree(
        lock,
        root,
        expected_generator_commit="a" * 40,
    )["passed"] is True
    target = root / "wikidata5m" / "wikidata5m_alias.tar.gz"
    payload = target.read_bytes()
    target.write_bytes(payload[:-1] + bytes([payload[-1] ^ 0x01]))
    with pytest.raises(ValueError, match="source byte drift"):
        verify_source_tree(lock, root, expected_generator_commit="a" * 40)


def test_cli_refuses_to_guess_a_metadata_transport(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--out", str(tmp_path / "plan.json")])
    assert error.value.code == 2


def test_cli_reports_every_unpinnable_binding_and_exits_nonzero(tmp_path, capsys):
    out = tmp_path / "plan.json"
    code = main(["--metadata-record", str(RECORD_PATH), "--out", str(out)])
    assert code == 1
    report = capsys.readouterr().out
    for source_id in source_lock_module._EXPECTED_SOURCE_IDS:
        assert source_id in report
    assert "UNPINNED" in report
    assert "finemath_selection" in report
    plan = SourcePinPlan.from_dict(json.loads(out.read_bytes()))
    assert plan.complete is False
    assert out.read_bytes() == plan.to_bytes()


def test_cli_allow_incomplete_still_writes_the_same_plan(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    assert main(["--metadata-record", str(RECORD_PATH), "--out", str(first)]) == 1
    assert (
        main(
            [
                "--metadata-record",
                str(RECORD_PATH),
                "--out",
                str(second),
                "--allow-incomplete",
            ]
        )
        == 0
    )
    assert first.read_bytes() == second.read_bytes()


def test_cli_records_the_evidence_it_consumed(tmp_path):
    out = tmp_path / "plan.json"
    evidence = tmp_path / "evidence.json"
    main(
        [
            "--metadata-record",
            str(RECORD_PATH),
            "--out",
            str(out),
            "--record",
            str(evidence),
            "--allow-incomplete",
        ]
    )
    replayed = resolve_source_pins(load_metadata_record(evidence))
    assert replayed.to_bytes() == out.read_bytes()


def test_cli_never_overwrites_an_existing_plan(tmp_path):
    out = tmp_path / "plan.json"
    out.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="already exists"):
        main(["--metadata-record", str(RECORD_PATH), "--out", str(out)])


def test_cli_egress_summary_names_only_the_hosts_it_contacts(tmp_path, capsys):
    main(
        [
            "--metadata-record",
            str(RECORD_PATH),
            "--out",
            str(tmp_path / "plan.json"),
            "--allow-incomplete",
        ]
    )
    report = capsys.readouterr().out
    assert "github.com" in report
    assert "huggingface.co" in report


def test_recorded_client_is_immune_to_caller_mutation(record):
    client = RecordedMetadataClient(record)
    snapshot = copy.deepcopy(record)
    record["git_ls_remote"].clear()
    assert client.git_ls_remote(
        "https://github.com/asaparov/prontoqa.git",
        "HEAD",
    ) == snapshot["git_ls_remote"]["https://github.com/asaparov/prontoqa.git"]["HEAD"]
