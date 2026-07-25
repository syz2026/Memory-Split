from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cluster.aws.corpus_builder.contracts import (
    CORPUS_BUCKET,
    CORPUS_KEY_PREFIX,
    phase_receipt_from_bytes,
)
from cluster.aws.corpus_builder.s3 import PublicationError
from corpusgen.parallel import FixtureRenderer, fixture_catalog
from test_aws_corpus_builder_s3 import FakeClientError, VersionedFakeS3

from cluster.aws.corpus_builder import driver


_KMS_ARN = (
    "arn:aws:kms:us-east-1:056956104102:"
    "key/01234567-89ab-cdef-0123-456789abcdef"
)


class RecordingRunner:
    def __init__(self, *, stop_before: str | None = None, before=None) -> None:
        self.phases: list[str] = []
        self.stop_before = stop_before
        self.before = before

    def run(self, phase: str, action):
        self.phases.append(phase)
        if self.stop_before == phase:
            raise RuntimeError(f"interrupted before {phase}")
        if self.before is not None:
            self.before(phase)
        return action()


class InterruptingUploadS3(VersionedFakeS3):
    def __init__(self) -> None:
        super().__init__()
        self.render_uploads = 0
        self.interrupted = False

    def put_object(self, **kwargs: object):
        metadata = kwargs.get("Metadata")
        if isinstance(metadata, dict) and metadata.get("phase") == "render-pack":
            self.render_uploads += 1
            if self.render_uploads == 2 and not self.interrupted:
                self.interrupted = True
                raise FakeClientError("interrupted during render upload")
        return super().put_object(**kwargs)


def _request(tmp_path: Path) -> driver.CorpusBuildRequest:
    source_lock = tmp_path / "source-lock.json"
    source_lock.write_bytes(b"fixture source lock\n")
    work_root = tmp_path / "work"
    work_root.mkdir(mode=0o700)
    return driver.CorpusBuildRequest(
        build_id="b" * 64,
        package_sha256="c" * 64,
        source_lock_path=source_lock,
        source_lock_sha256=hashlib.sha256(source_lock.read_bytes()).hexdigest(),
        work_root=work_root,
        output_root=tmp_path / "output",
        workers=3,
        shard_count=2,
        bucket=CORPUS_BUCKET,
        prefix=CORPUS_KEY_PREFIX,
        kms_key_arn=_KMS_ARN,
    )


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _install_fixture_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cleanroom_shards: list[bytes] | None = None,
) -> dict[str, object]:
    real_execute = driver._execute_phase
    state: dict[str, object] = {"render_roots": []}

    def production_inputs(context) -> driver.VerifiedProductionInputs:
        value = state.get("inputs")
        if isinstance(value, driver.VerifiedProductionInputs):
            return value
        sidecar = _write(
            context.request.work_root / "fixture-inputs" / "dense.bin",
            b"fixture-sidecar\n",
        )
        value = driver.VerifiedProductionInputs(
            catalog=fixture_catalog(3),
            renderer=FixtureRenderer(),
            sidecars={"dense": sidecar},
        )
        state["inputs"] = value
        return value

    def phase_action_root(context, phase: str) -> Path:
        action_root = getattr(driver, "_phase_action_root", None)
        if callable(action_root):
            return action_root(context, phase)
        return driver._phase_local_root(context.request, phase)

    def verify(corpus: Path, *, expected_build_id: str):
        receipt = json.loads((corpus / "receipt.json").read_bytes())
        assert receipt["build_id"] == expected_build_id
        shard = (corpus / "shards" / "00000.bin").read_bytes()
        if cleanroom_shards is not None and "cleanroom-" in corpus.name:
            cleanroom_shards.append(shard)
        return receipt

    def execute(context, phase: str):
        if phase == "cleanroom-verify":
            return real_execute(context, phase)

        if phase == "render-pack":
            root = phase_action_root(context, phase)
            render_roots = state["render_roots"]
            assert isinstance(render_roots, list)
            render_roots.append(root)
            receipt = (
                json.dumps(
                    {"build_id": context.request.build_id},
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
                + b"\n"
            )
            _write(root / "receipt.json", receipt)
            _write(
                root / "shards" / "00000.bin",
                b"original-corpus-shard\n",
            )
            artifacts = tuple(
                driver._Artifact(
                    path=path,
                    relative_key=(
                        "phase-artifacts/render-pack/"
                        f"{path.relative_to(root).as_posix()}"
                    ),
                )
                for path in sorted(
                    (
                        root / "receipt.json",
                        root / "shards" / "00000.bin",
                    )
                )
            )
            return driver._PhaseOutcome(artifacts=artifacts)

        if phase == "catalog":
            inputs = production_inputs(context)
            artifact = phase_action_root(context, phase) / "production-inputs.json"
            _write(
                artifact,
                (
                    json.dumps(
                        {
                            "catalog_sha256": inputs.catalog.sha256,
                            "renderer_id": inputs.renderer.renderer_id,
                            "sidecars": {
                                name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for name, path in inputs.sidecars.items()
                            },
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("ascii")
                    + b"\n"
                ),
            )
            return driver._PhaseOutcome(
                artifacts=(
                    driver._Artifact(
                        path=artifact,
                        relative_key=f"phase-artifacts/{phase}/{artifact.name}",
                    ),
                )
            )

        if phase == "local-verify":
            verify(
                context.request.output_root,
                expected_build_id=context.request.build_id,
            )

        if phase == "s3-publish":
            return driver._PhaseOutcome(
                artifacts=tuple(
                    driver._Artifact(
                        path=path,
                        relative_key=(
                            "corpus/"
                            f"{path.relative_to(context.request.output_root).as_posix()}"
                        ),
                    )
                    for path in sorted(
                        (
                            context.request.output_root / "receipt.json",
                            context.request.output_root / "shards" / "00000.bin",
                        )
                    )
                )
            )

        artifact = _write(
            phase_action_root(context, phase) / "phase-artifact.json",
            (
                json.dumps(
                    {"build_id": context.request.build_id, "phase": phase},
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
                + b"\n"
            ),
        )
        return driver._PhaseOutcome(
            artifacts=(
                driver._Artifact(
                    path=artifact,
                    relative_key=f"phase-artifacts/{phase}/{artifact.name}",
                ),
            )
        )

    monkeypatch.setattr(driver, "verify_parallel_corpus", verify)
    monkeypatch.setattr(driver, "_load_context_production_inputs", production_inputs)
    monkeypatch.setattr(driver, "_execute_phase", execute)
    return state


def _receipt_payload(s3: VersionedFakeS3, key_fragment: str) -> bytes:
    matches = [
        entry["BodyBytes"]
        for (_bucket, key), entries in s3._versions.items()
        if key_fragment in key
        for entry in entries
    ]
    assert len(matches) == 1
    payload = matches[0]
    assert isinstance(payload, bytes)
    return payload


def test_driver_runs_all_phases_and_publishes_final_receipt_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fixture_pipeline(monkeypatch)
    request = _request(tmp_path)
    s3 = VersionedFakeS3()
    runner = RecordingRunner()

    result = driver.run_corpus_build(request, s3=s3, runner=runner)

    assert result.uri.endswith("/receipts/final.json")
    assert runner.phases == list(driver.PHASES)
    put_keys = [str(call["Key"]) for call in s3.put_calls]
    assert put_keys[-1].endswith("/receipts/final.json")
    publish_receipt_index = next(
        index
        for index, key in enumerate(put_keys)
        if "/receipts/phases/05-s3-publish-" in key
    )
    corpus_indexes = [
        index
        for index, call in enumerate(s3.put_calls)
        if call["Metadata"].get("phase") == "s3-publish"
    ]
    assert corpus_indexes
    assert all(index < publish_receipt_index for index in corpus_indexes)
    cleanroom_receipt_index = next(
        index
        for index, key in enumerate(put_keys)
        if "/receipts/phases/06-cleanroom-verify-" in key
    )
    assert publish_receipt_index < cleanroom_receipt_index < len(put_keys) - 1


def test_driver_reuses_only_exact_verified_phase_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fixture_pipeline(monkeypatch)
    request = _request(tmp_path)
    s3 = VersionedFakeS3()
    interrupted = RecordingRunner(stop_before="render-pack")

    with pytest.raises(RuntimeError, match="interrupted"):
        driver.run_corpus_build(request, s3=s3, runner=interrupted)

    resumed = RecordingRunner()
    driver.run_corpus_build(request, s3=s3, runner=resumed)

    assert resumed.phases == list(driver.PHASES[3:])


def test_changed_shard_count_cannot_reuse_phase_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fixture_pipeline(monkeypatch)
    request = _request(tmp_path)
    s3 = VersionedFakeS3()
    with pytest.raises(RuntimeError, match="interrupted"):
        driver.run_corpus_build(
            request,
            s3=s3,
            runner=RecordingRunner(stop_before="render-pack"),
        )
    changed = replace(request, shard_count=request.shard_count + 1)
    resumed = RecordingRunner(stop_before="source-stage")

    with pytest.raises(RuntimeError, match="interrupted before source-stage"):
        driver.run_corpus_build(changed, s3=s3, runner=resumed)

    assert resumed.phases == ["source-stage"]


def test_reused_catalog_rejects_changed_production_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state = _install_fixture_pipeline(monkeypatch)
    request = _request(tmp_path)
    s3 = VersionedFakeS3()
    with pytest.raises(RuntimeError, match="interrupted"):
        driver.run_corpus_build(
            request,
            s3=s3,
            runner=RecordingRunner(stop_before="render-pack"),
        )
    inputs = state["inputs"]
    assert isinstance(inputs, driver.VerifiedProductionInputs)
    inputs.sidecars["dense"].write_bytes(b"changed-sidecar\n")
    resumed = RecordingRunner()

    with pytest.raises(PublicationError, match="catalog.*input"):
        driver.run_corpus_build(request, s3=s3, runner=resumed)

    assert resumed.phases == []


def test_interrupted_action_keeps_partial_output_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state = _install_fixture_pipeline(monkeypatch)
    request = _request(tmp_path)
    s3 = VersionedFakeS3()
    execute = driver._execute_phase
    interrupted = False

    def interrupt_after_output(context, phase: str):
        nonlocal interrupted
        outcome = execute(context, phase)
        if phase == "render-pack" and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted during render-pack action")
        return outcome

    monkeypatch.setattr(driver, "_execute_phase", interrupt_after_output)

    with pytest.raises(RuntimeError, match="during render-pack action"):
        driver.run_corpus_build(request, s3=s3, runner=RecordingRunner())

    render_roots = state["render_roots"]
    assert isinstance(render_roots, list)
    assert render_roots
    assert render_roots[0] != request.output_root
    assert not request.output_root.exists()

    resumed = RecordingRunner()
    driver.run_corpus_build(request, s3=s3, runner=resumed)

    assert resumed.phases == list(driver.PHASES[3:])
    assert len(render_roots) == 2
    assert len(set(render_roots)) == 2
    assert all(root != request.output_root for root in render_roots)


def test_interrupted_upload_reruns_in_a_fresh_private_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state = _install_fixture_pipeline(monkeypatch)
    request = _request(tmp_path)
    s3 = InterruptingUploadS3()
    first = RecordingRunner()

    with pytest.raises(PublicationError, match="upload failed"):
        driver.run_corpus_build(request, s3=s3, runner=first)

    assert first.phases == list(driver.PHASES[:4])
    assert s3.render_uploads == 2
    assert not any(
        "/receipts/phases/03-render-pack-" in str(call["Key"])
        for call in s3.put_calls
    )

    resumed = RecordingRunner()
    driver.run_corpus_build(request, s3=s3, runner=resumed)

    render_roots = state["render_roots"]
    assert isinstance(render_roots, list)
    assert resumed.phases == list(driver.PHASES[3:])
    assert len(render_roots) == 2
    assert len(set(render_roots)) == 2
    assert all(root != request.output_root for root in render_roots)


def test_successful_phase_discards_private_producer_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fixture_pipeline(monkeypatch)
    request = _request(tmp_path)
    s3 = VersionedFakeS3()
    execute = driver._execute_phase
    containers: list[Path] = []

    def leave_private_scratch(context, phase: str):
        outcome = execute(context, phase)
        if phase == "render-pack":
            container = context.attempt_container
            assert isinstance(container, Path)
            containers.append(container)
            _write(container / "producer-private" / "scratch.bin", b"scratch\n")
        return outcome

    monkeypatch.setattr(driver, "_execute_phase", leave_private_scratch)

    with pytest.raises(RuntimeError, match="interrupted before local-verify"):
        driver.run_corpus_build(
            request,
            s3=s3,
            runner=RecordingRunner(stop_before="local-verify"),
        )

    assert containers
    assert not containers[0].exists()
    assert request.output_root.is_dir()


def test_resume_rejects_missing_exact_object_before_running_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fixture_pipeline(monkeypatch)
    request = _request(tmp_path)
    s3 = VersionedFakeS3()
    with pytest.raises(RuntimeError, match="interrupted"):
        driver.run_corpus_build(
            request,
            s3=s3,
            runner=RecordingRunner(stop_before="wikidata-view"),
        )
    receipt = phase_receipt_from_bytes(
        _receipt_payload(s3, "/receipts/phases/00-source-stage-")
    )
    s3.mutate(receipt.objects[0], "missing-version")
    resumed = RecordingRunner()

    with pytest.raises(PublicationError, match="exact|version|receipt"):
        driver.run_corpus_build(request, s3=s3, runner=resumed)

    assert resumed.phases == []


def test_resume_rejects_local_dependency_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fixture_pipeline(monkeypatch)
    request = _request(tmp_path)
    s3 = VersionedFakeS3()
    with pytest.raises(RuntimeError, match="interrupted"):
        driver.run_corpus_build(
            request,
            s3=s3,
            runner=RecordingRunner(stop_before="wikidata-view"),
        )
    artifact = (
        request.work_root
        / "phases"
        / "source-stage"
        / "phase-artifact.json"
    )
    artifact.write_bytes(b"drift\n")
    resumed = RecordingRunner()

    with pytest.raises(PublicationError, match="local.*drift"):
        driver.run_corpus_build(request, s3=s3, runner=resumed)

    assert resumed.phases == []


def test_cleanroom_downloads_receipt_pinned_version_not_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cleanroom_shards: list[bytes] = []
    _install_fixture_pipeline(
        monkeypatch,
        cleanroom_shards=cleanroom_shards,
    )
    request = _request(tmp_path)
    s3 = VersionedFakeS3()

    def install_conflicting_latest(phase: str) -> None:
        if phase != "cleanroom-verify":
            return
        receipt = phase_receipt_from_bytes(
            _receipt_payload(s3, "/receipts/phases/05-s3-publish-")
        )
        shard = next(obj for obj in receipt.objects if obj.uri.endswith(".bin"))
        key = shard.uri.split(f"s3://{request.bucket}/", 1)[1]
        s3.install(
            bucket=request.bucket,
            key=key,
            payload=b"foreign-latest-version\n",
            kms_key_arn=request.kms_key_arn,
            metadata={"sha256": hashlib.sha256(b"foreign-latest-version\n").hexdigest()},
        )

    driver.run_corpus_build(
        request,
        s3=s3,
        runner=RecordingRunner(before=install_conflicting_latest),
    )

    assert cleanroom_shards == [b"original-corpus-shard\n"]


def test_final_receipt_rejects_cleanroom_object_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fixture_pipeline(monkeypatch)
    request = _request(tmp_path)
    s3 = VersionedFakeS3()
    real_execute = driver._execute_phase

    def disagree(context, phase: str):
        outcome = real_execute(context, phase)
        if phase != "cleanroom-verify":
            return outcome
        first, *remaining = outcome.objects
        return driver._PhaseOutcome(
            objects=(replace(first, bytes=first.bytes + 1), *remaining)
        )

    monkeypatch.setattr(driver, "_execute_phase", disagree)

    with pytest.raises(PublicationError, match="clean-room.*agree"):
        driver.run_corpus_build(
            request,
            s3=s3,
            runner=RecordingRunner(),
        )

    assert not any(
        str(call["Key"]).endswith("/receipts/final.json")
        for call in s3.put_calls
    )


def test_verified_production_factory_verifies_authorities_before_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, object]] = []
    lock = SimpleNamespace(generator_commit="a" * 40, sha256="d" * 64)
    built = SimpleNamespace(root=tmp_path / "derived" / "wikidata" / ("e" * 64))
    view = SimpleNamespace(
        receipt=SimpleNamespace(
            source_lock_sha256=lock.sha256,
            generator_commit=lock.generator_commit,
        )
    )
    catalog = fixture_catalog(3)
    renderer = FixtureRenderer()
    sidecar = _write(tmp_path / "dense.bin", b"\x01")

    def load_lock(path, *, expected_generator_commit):
        calls.append(("load-lock", (path, expected_generator_commit)))
        return lock

    def verify_tree(actual_lock, root, *, expected_generator_commit):
        calls.append(
            ("verify-tree", (actual_lock, root, expected_generator_commit))
        )
        return {}

    def build_view(lock_path, root, derived, *, expected_generator_commit):
        calls.append(
            (
                "build-view",
                (lock_path, root, derived, expected_generator_commit),
            )
        )
        return built

    def verify_view(lock_path, root, derived, *, expected_generator_commit):
        calls.append(
            (
                "verify-view",
                (lock_path, root, derived, expected_generator_commit),
            )
        )
        return view

    def adapter(**kwargs):
        calls.append(("adapter", kwargs))
        return driver.VerifiedProductionInputs(
            catalog=catalog,
            renderer=renderer,
            sidecars={"dense": sidecar},
        )

    monkeypatch.setattr(driver, "load_source_lock", load_lock)
    monkeypatch.setattr(driver, "verify_source_tree", verify_tree)
    monkeypatch.setattr(driver, "build_wikidata_derived_view", build_view)
    monkeypatch.setattr(driver, "verify_wikidata_derived_view", verify_view)
    monkeypatch.setattr(driver, "_production_inputs_from_verified_view", adapter)

    result = driver.load_verified_production_inputs(
        source_lock_path=tmp_path / "source-lock.json",
        source_root=tmp_path / "sources",
        derived_root=tmp_path / "derived",
        expected_generator_commit="a" * 40,
    )

    assert result.catalog is catalog
    assert result.renderer is renderer
    assert dict(result.sidecars) == {"dense": sidecar}
    assert [name for name, _value in calls] == [
        "load-lock",
        "verify-tree",
        "build-view",
        "verify-view",
        "adapter",
    ]
    with pytest.raises(TypeError):
        result.sidecars["other"] = sidecar  # type: ignore[index]
