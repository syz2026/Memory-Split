from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


class _QueueAwsRunner:
    def __init__(self) -> None:
        self.outputs: list[object] = []
        self.calls: list[tuple[list[str], str]] = []

    def run_json(self, argv, *, operation: str):
        self.calls.append((list(argv), operation))
        if not self.outputs:
            raise AssertionError(f"unexpected AWS call: {operation}")
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def _v3_pair(
    tmp_path: Path,
    *,
    status: str = "Pending",
    command_id: str | None = "cmd-0123456789abcdef0",
):
    from cluster.aws.p5.profile import load_aws_p5_profile
    from msctl.aws_p5 import AwsP5Backend
    from msctl.contracts import load_run_manifest
    from msctl.jsonutil import canonical_json
    from msctl.state import StateStore
    from tests.test_aws_canary import (
        BOOT_ID,
        IMAGE,
        IMAGE_DIGEST,
        INSTANCE_ID,
        _case,
        _load_module,
    )

    case = _case(tmp_path / "controller", _load_module())
    manifest = load_run_manifest(
        case["manifest_path"],
        repo_root=case["release_root"],
    )
    release = SimpleNamespace(
        provider="aws-p5.48xlarge",
        archive_sha256=manifest.release_sha256,
        receipt_sha256=manifest.release_receipt_sha256,
        members_sha256="a" * 64,
        source_commit=manifest.source_commit,
        source_tree=manifest.source_tree,
    )
    runtime = SimpleNamespace(
        region="us-east-1",
        s3_root="s3://memorysplit-prod/confirmatory-v3",
        ami_id="ami-0123456789abcdef0",
        container_image=IMAGE,
        container_digest=IMAGE_DIGEST,
        uid=1000,
        gid=1000,
    )
    runner = _QueueAwsRunner()
    state_root = tmp_path / "state"
    backend = AwsP5Backend(
        profile=load_aws_p5_profile(
            case["release_root"]
            / "cluster"
            / "profiles"
            / "aws-p5.48xlarge-v3.json"
        ),
        runtime=runtime,
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=state_root,
        runner=runner,
        approval_verifier=lambda **_kwargs: {},
        corpus_verifier=case["dataset_verifier"],
        identity_verifier=lambda *_args: True,
        environ={},
    )
    evidence = {
        "dataset_pointer_sha256": manifest.dataset_pointer_sha256,
        "dataset_verification_sha256": manifest.dataset_receipt_sha256,
        "environment_receipt_sha256": "e" * 64,
        "instance_id": INSTANCE_ID,
        "boot_id": BOOT_ID,
    }
    terminate_at = (
        datetime.now(timezone.utc).replace(microsecond=0)
        + timedelta(hours=1)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    envelope = backend._operation_envelope(
        backend._training_operation_intent(
            operation="submit",
            release=release,
            manifest=manifest,
            terminate_at=terminate_at,
            evidence=evidence,
        ),
        instance_id=INSTANCE_ID,
        terminate_at=terminate_at,
    )
    digest = hashlib.sha256(canonical_json(envelope)).hexdigest()
    states = []
    for run in manifest.runs:
        state = backend._new_aws_run_state(
            run=run,
            manifest=manifest,
            operation="submit",
            instance_id=INSTANCE_ID,
            terminate_at=terminate_at,
            intent=envelope,
            published={
                "intent_sha256": digest,
                "intent_uri": (
                    f"{runtime.s3_root}/operations/intents/"
                    f"sha256/{digest}.json"
                ),
            },
            attempt=1,
        )
        state["command_id"] = command_id
        state["send_attempted"] = True
        state["status"] = status
        states.append(state)
    store = StateStore(state_root)
    with store.locked():
        backend._write_paired_states(store, manifest, states)
    selected = {
        "instance_id": INSTANCE_ID,
        "instance_type": "p5.48xlarge",
        "state": "running",
        "instance_profile_arn": (
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        "ami_id": runtime.ami_id,
        **backend._selected_binding(manifest, terminate_at=terminate_at),
    }
    return SimpleNamespace(
        backend=backend,
        evidence=evidence,
        envelope=envelope,
        instance_id=INSTANCE_ID,
        manifest=manifest,
        release=release,
        runner=runner,
        selected=selected,
        state_root=state_root,
        states=states,
        store=store,
        terminate_at=terminate_at,
    )


def _v2_pair(
    tmp_path: Path,
    *,
    status: str = "Success",
    journal: bool,
):
    from msctl.aws_p5 import AwsP5Backend
    from msctl.state import StateStore
    from tests.test_msctl import (
        _FakeAwsRunner,
        _aws_manifest_object,
        _aws_profile_object,
        _aws_release_object,
        _aws_run_state,
        _aws_runtime_object,
    )

    manifest = _aws_manifest_object()
    release = _aws_release_object()
    runner = _FakeAwsRunner()
    state_root = tmp_path / "state"
    backend = AwsP5Backend(
        profile=_aws_profile_object(),
        runtime=_aws_runtime_object(),
        instance_profile_arn=(
            "arn:aws:iam::123456789012:instance-profile/memorysplit-p5"
        ),
        state_root=state_root,
        runner=runner,
        approval_verifier=lambda **_kwargs: {},
    )
    states = [
        _aws_run_state(
            backend,
            release,
            manifest,
            run,
            status=status,
            command_id="cmd-0123456789abcdef0",
        )
        for run in manifest.runs
    ]
    store = StateStore(state_root)
    with store.locked():
        if journal:
            backend._write_paired_states(store, manifest, states)
        else:
            for state in states:
                store.write_run(str(state["run_id"]), state)
    return SimpleNamespace(
        backend=backend,
        manifest=manifest,
        release=release,
        runner=runner,
        state_root=state_root,
        states=states,
        store=store,
    )


def _generation_paths(fixture) -> list[Path]:
    return [
        fixture.state_root
        / "intents"
        / f"aws-{fixture.manifest.sha256}.json",
        *sorted(
            fixture.state_root / "runs" / f"{run.run_id}.json"
            for run in fixture.manifest.runs
        ),
    ]


def _generation_bytes(fixture) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in _generation_paths(fixture)}


def _read_generation(fixture):
    with fixture.store.locked():
        journal = fixture.store.read_aws_pair(fixture.manifest.sha256)
        runs = {
            run.run_id: fixture.store.read_run(run.run_id)
            for run in fixture.manifest.runs
        }
    assert journal is not None
    return journal, runs


def _journal(fixture, states: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "provider": "aws-p5.48xlarge",
        "run_manifest_sha256": fixture.manifest.sha256,
        "operation_id": states[0]["operation_id"],
        "states": copy.deepcopy(states),
    }


def _transition_to_resume(fixture) -> list[dict[str, object]]:
    proposed = copy.deepcopy(fixture.states)
    checkpoint_objects = [
        {
            "arm": arm,
            "bytes": 123,
            "sha256": digest * 64,
            "uri": f"s3://bucket/{arm}.pt",
            "version_id": f"{arm}-version-1",
        }
        for arm, digest in (("dense", "1"), ("split90", "2"))
    ]
    for state in proposed:
        state.update(
            {
                "operation": "resume",
                "operation_id": "3" * 64,
                "intent_sha256": "4" * 64,
                "intent_uri": "s3://bucket/resume-intent.json",
                "command_id": None,
                "status": "INTENT_PUBLISHED",
                "attempt": 2,
                "send_attempted": False,
                "checkpoint_receipt": {
                    "sha256": "5" * 64,
                    "uri": "s3://bucket/receipt.json",
                    "version_id": "receipt-version-1",
                },
                "checkpoint_objects": copy.deepcopy(checkpoint_objects),
                "prior_command_ids": ["cmd-0123456789abcdef0"],
                "updated_at": "2099-01-02T03:04:05Z",
            }
        )
    with fixture.store.locked():
        fixture.backend._write_paired_states(
            fixture.store,
            fixture.manifest,
            proposed,
        )
    fixture.states = proposed
    return proposed


def test_v3_exact_paired_refresh_does_not_touch_timestamp_inputs_or_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msctl.aws_p5 as aws_p5

    fixture = _v3_pair(tmp_path)
    before_bytes = _generation_bytes(fixture)
    supplied = copy.deepcopy(list(reversed(fixture.states)))
    before_supplied = copy.deepcopy(supplied)

    def timestamp_must_not_be_read() -> str:
        raise AssertionError("exact replay must not request a timestamp")

    monkeypatch.setattr(aws_p5, "_timestamp", timestamp_must_not_be_read)
    with fixture.store.locked():
        refreshed = fixture.backend._refresh_paired_states(
            fixture.store,
            fixture.manifest,
            supplied,
            {"status": "Pending"},
        )

    assert refreshed == supplied
    assert supplied == before_supplied
    assert _generation_bytes(fixture) == before_bytes


def test_v3_changed_refresh_binds_reversed_states_by_id_with_one_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msctl.aws_p5 as aws_p5

    fixture = _v3_pair(tmp_path)
    supplied = copy.deepcopy(list(reversed(fixture.states)))
    before_supplied = copy.deepcopy(supplied)
    now = "2099-01-02T03:04:05Z"
    monkeypatch.setattr(aws_p5, "_timestamp", lambda: now)

    with fixture.store.locked():
        refreshed = fixture.backend._refresh_paired_states(
            fixture.store,
            fixture.manifest,
            supplied,
            {"status": "InProgress"},
        )

    assert supplied == before_supplied
    assert {state["run_id"] for state in refreshed} == {
        run.run_id for run in fixture.manifest.runs
    }
    journal, runs = _read_generation(fixture)
    journal_by_id = {state["run_id"]: state for state in journal["states"]}
    assert journal_by_id == runs
    assert {state["status"] for state in runs.values()} == {"InProgress"}
    assert {state["updated_at"] for state in runs.values()} == {now}
    for run_id, state in runs.items():
        assert state["run_id"] == run_id


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_v3_paired_refresh_restores_exact_prior_bytes_after_each_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_at: int,
) -> None:
    import msctl.aws_p5 as aws_p5
    import msctl.fsutil as fsutil
    import msctl.state as state_module

    fixture = _v3_pair(tmp_path)
    before_bytes = _generation_bytes(fixture)
    supplied = copy.deepcopy(fixture.states)
    before_supplied = copy.deepcopy(supplied)
    calls = 0
    original_json_write = state_module.atomic_write_json_at
    original_byte_write = fsutil.atomic_write_at

    def fail_json_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError(f"injected JSON install failure {fail_at}")
        return original_json_write(*args, **kwargs)

    def fail_byte_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError(f"injected byte install failure {fail_at}")
        return original_byte_write(*args, **kwargs)

    monkeypatch.setattr(state_module, "atomic_write_json_at", fail_json_write)
    monkeypatch.setattr(
        state_module,
        "atomic_write_at",
        fail_byte_write,
        raising=False,
    )
    monkeypatch.setattr(aws_p5, "_timestamp", lambda: "2099-01-02T03:04:05Z")

    with fixture.store.locked(), pytest.raises(Exception, match="injected|state"):
        fixture.backend._refresh_paired_states(
            fixture.store,
            fixture.manifest,
            supplied,
            {"status": "InProgress"},
        )

    assert supplied == before_supplied
    assert _generation_bytes(fixture) == before_bytes


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_v3_paired_refresh_rolls_back_failures_after_each_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_at: int,
) -> None:
    import msctl.aws_p5 as aws_p5
    import msctl.fsutil as fsutil
    import msctl.state as state_module

    fixture = _v3_pair(tmp_path)
    before_bytes = _generation_bytes(fixture)
    calls = 0
    original_write = fsutil.atomic_write_at

    def install_then_fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        original_write(*args, **kwargs)
        if calls == fail_at:
            raise OSError(f"injected post-install failure {fail_at}")

    monkeypatch.setattr(state_module, "atomic_write_at", install_then_fail)
    monkeypatch.setattr(aws_p5, "_timestamp", lambda: "2099-01-02T03:04:05Z")
    with fixture.store.locked(), pytest.raises(Exception, match="state"):
        fixture.backend._refresh_paired_states(
            fixture.store,
            fixture.manifest,
            copy.deepcopy(fixture.states),
            {"status": "InProgress"},
        )

    assert _generation_bytes(fixture) == before_bytes


@pytest.mark.parametrize(
    "mutation",
    ["command", "operation", "provenance"],
)
def test_v3_transaction_rejects_binding_drift_before_any_install(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _v3_pair(tmp_path)
    proposed = copy.deepcopy(fixture.states)
    if mutation == "command":
        for state in proposed:
            state["command_id"] = "cmd-99999999999999999"
    elif mutation == "provenance":
        for state in proposed:
            state["release_receipt_sha256"] = "f" * 64
    else:
        checkpoint_objects = [
            {
                "arm": arm,
                "bytes": 123,
                "sha256": digest * 64,
                "uri": f"s3://bucket/{arm}.pt",
                "version_id": f"{arm}-version-1",
            }
            for arm, digest in (("dense", "1"), ("split90", "2"))
        ]
        for state in proposed:
            state["operation"] = "resume"
            state["checkpoint_receipt"] = {
                "sha256": "3" * 64,
                "uri": "s3://bucket/receipt.json",
                "version_id": "receipt-version-1",
            }
            state["checkpoint_objects"] = copy.deepcopy(checkpoint_objects)
            state["prior_command_ids"] = ["cmd-0123456789abcdef0"]
    before_bytes = _generation_bytes(fixture)
    before_proposed = copy.deepcopy(proposed)

    with fixture.store.locked(), pytest.raises(Exception):
        fixture.store.write_aws_pair_transaction(
            fixture.manifest.sha256,
            _journal(fixture, proposed),
            proposed,
        )

    assert proposed == before_proposed
    assert _generation_bytes(fixture) == before_bytes


@pytest.mark.parametrize(
    "mutation",
    ["run_id", "arm", "command", "schema", "provider", "manifest"],
)
def test_v3_initial_pair_requires_exact_manifest_set_and_uniform_bindings(
    tmp_path: Path,
    mutation: str,
) -> None:
    from msctl.state import StateStore

    fixture = _v3_pair(tmp_path / "source")
    proposed = copy.deepcopy(fixture.states)
    if mutation == "run_id":
        proposed[1]["run_id"] = "memorysplit-v3-360m-s0-decoy"
    elif mutation == "arm":
        proposed[1]["arm"] = "dense"
    elif mutation == "command":
        proposed[1]["command_id"] = "cmd-99999999999999999"
    elif mutation == "schema":
        proposed[1]["schema_version"] = 1
    elif mutation == "provider":
        proposed[1]["provider"] = "other-provider"
    else:
        proposed[1]["run_manifest_sha256"] = "f" * 64
    store = StateStore(tmp_path / "target")

    with store.locked(), pytest.raises(Exception):
        fixture.backend._write_paired_states(
            store,
            fixture.manifest,
            proposed,
        )

    assert not list((tmp_path / "target" / "runs").glob("*.json"))
    assert not list((tmp_path / "target" / "intents").glob("*.json"))


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_v3_initial_pair_failure_removes_every_new_generation_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_at: int,
) -> None:
    import msctl.fsutil as fsutil
    import msctl.state as state_module
    from msctl.state import StateStore

    fixture = _v3_pair(tmp_path / "source")
    store = StateStore(tmp_path / "target")
    calls = 0
    original_write = fsutil.atomic_write_at

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError(f"injected initial install failure {fail_at}")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(state_module, "atomic_write_at", fail_once)
    with store.locked(), pytest.raises(Exception, match="injected|state"):
        fixture.backend._write_paired_states(
            store,
            fixture.manifest,
            copy.deepcopy(fixture.states),
        )

    assert not list((tmp_path / "target" / "runs").glob("*.json"))
    assert not list((tmp_path / "target" / "intents").glob("*.json"))


def test_v3_failed_rollback_poison_is_observed_before_lifecycle_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msctl.aws_p5 as aws_p5
    import msctl.fsutil as fsutil
    import msctl.state as state_module

    fixture = _v3_pair(tmp_path)
    calls = 0
    original_write = fsutil.atomic_write_at

    def fail_install_and_rollback(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls in {2, 3}:
            raise OSError(f"injected transaction failure {calls}")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(
        state_module,
        "atomic_write_at",
        fail_install_and_rollback,
    )
    monkeypatch.setattr(aws_p5, "_timestamp", lambda: "2099-01-02T03:04:05Z")
    with fixture.store.locked(), pytest.raises(Exception) as caught:
        fixture.backend._refresh_paired_states(
            fixture.store,
            fixture.manifest,
            copy.deepcopy(fixture.states),
            {"status": "InProgress"},
        )
    assert getattr(caught.value, "code", None) == "STATE_ROLLBACK_FAILED"

    monkeypatch.setattr(state_module, "atomic_write_at", original_write)
    with fixture.store.locked(), pytest.raises(Exception):
        fixture.backend._paired_states(fixture.store, fixture.manifest)


def test_v3_rollback_marker_failure_still_returns_fail_closed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msctl.aws_p5 as aws_p5
    import msctl.fsutil as fsutil
    import msctl.state as state_module

    fixture = _v3_pair(tmp_path)
    writes = 0
    original_write = fsutil.atomic_write_at
    original_open = state_module.os.open

    def fail_install_and_restore(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes in {2, 3}:
            raise OSError(f"injected write failure {writes}")
        return original_write(*args, **kwargs)

    def fail_marker_open(path, *args, **kwargs):
        if str(path).endswith(".rollback-failed"):
            raise OSError("injected rollback marker failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(state_module, "atomic_write_at", fail_install_and_restore)
    monkeypatch.setattr(state_module.os, "open", fail_marker_open)
    monkeypatch.setattr(aws_p5, "_timestamp", lambda: "2099-01-02T03:04:05Z")

    with fixture.store.locked(), pytest.raises(Exception) as caught:
        fixture.backend._refresh_paired_states(
            fixture.store,
            fixture.manifest,
            copy.deepcopy(fixture.states),
            {"status": "InProgress"},
        )

    assert getattr(caught.value, "code", None) == "STATE_ROLLBACK_FAILED"


def test_v3_validates_every_new_state_before_first_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msctl.state as state_module

    fixture = _v3_pair(tmp_path)
    proposed = copy.deepcopy(fixture.states)
    proposed[1]["status"] = "not-an-aws-status"
    installs = 0

    def record_install(*_args, **_kwargs):
        nonlocal installs
        installs += 1

    monkeypatch.setattr(state_module, "atomic_write_at", record_install)
    with fixture.store.locked(), pytest.raises(Exception):
        fixture.store.write_aws_pair_transaction(
            fixture.manifest.sha256,
            _journal(fixture, proposed),
            proposed,
        )

    assert installs == 0


def test_v3_existing_journal_divergence_fails_closed_without_repair(
    tmp_path: Path,
) -> None:
    from msctl.jsonutil import canonical_json

    fixture = _v3_pair(tmp_path)
    before_runs = {
        path: path.read_bytes() for path in _generation_paths(fixture)[1:]
    }
    journal_path = _generation_paths(fixture)[0]
    journal = json.loads(journal_path.read_text())
    journal["states"][0]["status"] = "Success"
    journal_path.write_bytes(canonical_json(journal) + b"\n")
    divergent_bytes = journal_path.read_bytes()

    with fixture.store.locked(), pytest.raises(Exception):
        fixture.backend._paired_states(fixture.store, fixture.manifest)

    assert journal_path.read_bytes() == divergent_bytes
    assert {
        path: path.read_bytes() for path in _generation_paths(fixture)[1:]
    } == before_runs


def test_v3_missing_journal_with_existing_runs_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _v3_pair(tmp_path)
    journal_path = _generation_paths(fixture)[0]
    journal_path.unlink()
    before_runs = {
        path: path.read_bytes() for path in _generation_paths(fixture)[1:]
    }

    with fixture.store.locked(), pytest.raises(Exception):
        fixture.backend._repair_paired_states(
            fixture.store,
            fixture.manifest,
        )

    assert not journal_path.exists()
    assert {
        path: path.read_bytes() for path in _generation_paths(fixture)[1:]
    } == before_runs


@pytest.mark.parametrize("version_field", ["receipt", "checkpoint"])
def test_v3_resume_version_drift_is_rejected_before_install(
    tmp_path: Path,
    version_field: str,
) -> None:
    fixture = _v3_pair(tmp_path)
    _transition_to_resume(fixture)
    proposed = copy.deepcopy(fixture.states)
    if version_field == "receipt":
        for state in proposed:
            state["checkpoint_receipt"]["version_id"] = "receipt-version-2"
    else:
        for state in proposed:
            state["checkpoint_objects"][0]["version_id"] = "dense-version-2"
    before_bytes = _generation_bytes(fixture)

    with fixture.store.locked(), pytest.raises(Exception):
        fixture.store.write_aws_pair_transaction(
            fixture.manifest.sha256,
            _journal(fixture, proposed),
            proposed,
        )

    assert _generation_bytes(fixture) == before_bytes


def test_v3_resume_refresh_and_exact_replay_keep_one_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msctl.aws_p5 as aws_p5

    fixture = _v3_pair(tmp_path)
    resume_states = _transition_to_resume(fixture)
    monkeypatch.setattr(aws_p5, "_timestamp", lambda: "2099-02-03T04:05:06Z")
    with fixture.store.locked():
        sending = fixture.backend._refresh_paired_states(
            fixture.store,
            fixture.manifest,
            resume_states,
            {"send_attempted": True, "status": "SENDING"},
        )
        recovered = fixture.backend._refresh_paired_states(
            fixture.store,
            fixture.manifest,
            sending,
            {
                "command_id": "cmd-resume-12345678",
                "status": "InProgress",
            },
        )
    fixture.states = recovered
    before_replay = _generation_bytes(fixture)

    monkeypatch.setattr(aws_p5, "_timestamp", lambda: "2099-03-04T05:06:07Z")
    with fixture.store.locked():
        repeated = fixture.backend._refresh_paired_states(
            fixture.store,
            fixture.manifest,
            copy.deepcopy(recovered),
            {
                "command_id": "cmd-resume-12345678",
                "status": "InProgress",
            },
        )

    assert repeated == recovered
    assert _generation_bytes(fixture) == before_replay
    journal, runs = _read_generation(fixture)
    assert {state["run_id"]: state for state in journal["states"]} == runs


def test_v3_submit_lost_response_recovery_and_exact_replay_keep_one_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msctl.aws_p5 as aws_p5
    from msctl.errors import MsctlError

    fixture = _v3_pair(
        tmp_path,
        status="SENDING",
        command_id=None,
    )
    recovered_command = "cmd-recovered-12345678"
    fixture.runner.outputs.extend(
        [
            MsctlError("AWS_COMMAND_FAILED", "started receipt absent"),
            MsctlError("AWS_COMMAND_FAILED", "terminal receipt absent"),
            {
                "commands": [
                    {
                        "command_id": recovered_command,
                        "status": "InProgress",
                        "comment": fixture.envelope["operation_id"],
                        "instance_ids": [fixture.instance_id],
                    }
                ]
            },
        ]
    )

    recovered = fixture.backend.submit(
        release=fixture.release,
        manifest=fixture.manifest,
        instance_id=fixture.instance_id,
        terminate_at=fixture.terminate_at,
        approval_path=tmp_path / "approval.json",
        apply=True,
        evidence=fixture.evidence,
    )

    assert recovered["idempotent"] is True
    assert recovered["command_id"] == recovered_command
    journal, runs = _read_generation(fixture)
    assert {state["run_id"]: state for state in journal["states"]} == runs
    assert {state["command_id"] for state in runs.values()} == {
        recovered_command
    }

    before_replay = _generation_bytes(fixture)
    monkeypatch.setattr(aws_p5, "_timestamp", lambda: "2099-02-03T04:05:06Z")
    fixture.runner.outputs.append(
        {
            "command": {
                "command_id": recovered_command,
                "status": "InProgress",
            }
        }
    )
    replayed = fixture.backend.submit(
        release=fixture.release,
        manifest=fixture.manifest,
        instance_id=fixture.instance_id,
        terminate_at=fixture.terminate_at,
        approval_path=tmp_path / "approval.json",
        apply=True,
        evidence=fixture.evidence,
    )

    assert replayed["idempotent"] is True
    assert _generation_bytes(fixture) == before_replay


def test_v3_authoritative_status_refresh_keeps_journal_and_runs_equal(
    tmp_path: Path,
) -> None:
    fixture = _v3_pair(tmp_path)
    fixture.runner.outputs.append(
        {
            "command": {
                "command_id": "cmd-0123456789abcdef0",
                "status": "InProgress",
            }
        }
    )

    observed = fixture.backend.status(
        release=fixture.release,
        manifest=fixture.manifest,
        cached=False,
    )

    assert observed["status"] == "InProgress"
    journal, runs = _read_generation(fixture)
    assert {state["run_id"]: state for state in journal["states"]} == runs
    assert {state["status"] for state in runs.values()} == {"InProgress"}


def test_v3_cancel_transition_keeps_journal_and_runs_equal(
    tmp_path: Path,
) -> None:
    fixture = _v3_pair(tmp_path, status="InProgress")
    fixture.runner.outputs.extend(
        [
            {
                "command": {
                    "command_id": "cmd-0123456789abcdef0",
                    "status": "InProgress",
                }
            },
            {},
        ]
    )

    cancelled = fixture.backend.cancel(
        release=fixture.release,
        manifest=fixture.manifest,
        approval_path=tmp_path / "approval.json",
        apply=True,
    )

    assert cancelled["cancelled"] == 1
    journal, runs = _read_generation(fixture)
    assert {state["run_id"]: state for state in journal["states"]} == runs
    assert {state["status"] for state in runs.values()} == {"Cancelling"}


def test_v3_cleanup_transition_keeps_journal_and_runs_equal(
    tmp_path: Path,
) -> None:
    fixture = _v3_pair(tmp_path, status="Failed")
    fixture.runner.outputs.extend(
        [
            {"instances": [fixture.selected]},
            {
                "attribute": {
                    "instance_id": fixture.instance_id,
                    "shutdown_behavior": "terminate",
                }
            },
            {
                "command": {
                    "command_id": "cmd-0123456789abcdef0",
                    "status": "Failed",
                }
            },
            {
                "terminating_instances": [
                    {
                        "instance_id": fixture.instance_id,
                        "current_state": "shutting-down",
                        "previous_state": "running",
                    }
                ]
            },
        ]
    )

    cleaned = fixture.backend.cleanup(
        release=fixture.release,
        manifest=fixture.manifest,
        approval_path=tmp_path / "approval.json",
        apply=True,
    )

    assert cleaned["terminated"] == 1
    journal, runs = _read_generation(fixture)
    assert {state["run_id"]: state for state in journal["states"]} == runs
    assert {state["status"] for state in runs.values()} == {"Terminating"}


def test_v2_journal_free_refresh_stays_journal_free_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msctl.aws_p5 as aws_p5

    fixture = _v2_pair(tmp_path, status="Pending", journal=False)
    monkeypatch.setattr(aws_p5, "_timestamp", lambda: "2099-01-02T03:04:05Z")
    with fixture.store.locked():
        refreshed = fixture.backend._refresh_paired_states(
            fixture.store,
            fixture.manifest,
            list(reversed(copy.deepcopy(fixture.states))),
            {"status": "InProgress"},
        )
        assert fixture.store.read_aws_pair(fixture.manifest.sha256) is None
        runs = {
            run.run_id: fixture.store.read_run(run.run_id)
            for run in fixture.manifest.runs
        }
    before_replay = {
        path: path.read_bytes()
        for path in sorted((fixture.state_root / "runs").glob("*.json"))
    }

    monkeypatch.setattr(aws_p5, "_timestamp", lambda: "2099-02-03T04:05:06Z")
    with fixture.store.locked():
        replayed = fixture.backend._refresh_paired_states(
            fixture.store,
            fixture.manifest,
            copy.deepcopy(refreshed),
            {"status": "InProgress"},
        )

    assert replayed == refreshed
    assert {state["status"] for state in runs.values()} == {"InProgress"}
    assert {
        path: path.read_bytes()
        for path in sorted((fixture.state_root / "runs").glob("*.json"))
    } == before_replay


def test_v2_evaluate_leaves_existing_pair_journal_equal_to_runs(
    tmp_path: Path,
) -> None:
    import base64

    from msctl.aws_p5 import ARGV_DOCUMENT_NAME, ARGV_DOCUMENT_SHA256
    from msctl.jsonutil import canonical_json
    from tests.test_msctl import _selected_instance

    fixture = _v2_pair(tmp_path, journal=True)
    instance_id = "i-0123456789abcdef0"
    intent = fixture.backend._operation_envelope(
        fixture.backend._evaluation_operation_intent(
            fixture.release,
            fixture.manifest,
        ),
        instance_id=instance_id,
        terminate_at=str(fixture.states[0]["terminate_at"]),
    )
    payload = canonical_json(intent)
    digest = hashlib.sha256(payload).hexdigest()
    checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    fixture.runner.outputs.extend(
        [
            {"instances": [_selected_instance(fixture.manifest, bound=True)]},
            {
                "attribute": {
                    "instance_id": instance_id,
                    "shutdown_behavior": "terminate",
                }
            },
            {
                "command": {
                    "command_id": "cmd-0123456789abcdef0",
                    "status": "Success",
                }
            },
            {
                "managed_instances": [
                    {
                        "instance_id": instance_id,
                        "ping_status": "Online",
                    }
                ]
            },
            {
                "documents": [
                    {
                        "name": ARGV_DOCUMENT_NAME,
                        "hash": ARGV_DOCUMENT_SHA256,
                        "status": "Active",
                    }
                ]
            },
            {
                "object": {
                    "checksum_sha256": checksum,
                    "version_id": "evaluation-version-1",
                }
            },
            {
                "object": {
                    "checksum_sha256": checksum,
                    "content_length": len(payload),
                    "metadata": {
                        "operation-id": intent["operation_id"],
                        "sha256": digest,
                    },
                    "version_id": "evaluation-version-1",
                }
            },
            {"command": {"command_id": "cmd-evaluate-12345678"}},
        ]
    )

    evaluated = fixture.backend.evaluate(
        release=fixture.release,
        manifest=fixture.manifest,
        approval_path=tmp_path / "approval.json",
        apply=True,
    )

    assert evaluated["submitted"] == 1
    with fixture.store.locked():
        journal = fixture.store.read_aws_pair(fixture.manifest.sha256)
        runs = {
            run.run_id: fixture.store.read_run(run.run_id)
            for run in fixture.manifest.runs
        }
    assert journal is not None
    assert {state["run_id"]: state for state in journal["states"]} == runs
