from __future__ import annotations

import hashlib
import importlib
import inspect
import io
import json
import os
import pickle
from dataclasses import replace
from pathlib import Path

import pytest

import evals.reasoning_v3.aws_authority as authority_module
import evals.reasoning_v3.runner as runner_module
from corpusgen.parallel.canonical import canonical_json_bytes
from evals.reasoning_v3.aws_authority import (
    EVALUATOR_ROLE_ARN,
    S3ObjectVersion,
    VerifiedAwsAuthority,
)
from evals.reasoning_v3.contracts import FAMILY_ORDER
from evals.reasoning_v3.runner import (
    REQUIRED_VALIDITY_GATES,
    CheckpointBinding,
    CodeBinding,
    GateEvidence,
    RawPrediction,
    ReleaseBinding,
    RunnerError,
    _build_checkpoint_result,
    _checkpoint_result_key,
    _greedy_decode_model,
    _publish_checkpoint_result,
    _score_exact_output,
    _validate_checkpoint_result,
    run_frozen_checkpoint_evaluation,
    validate_frozen_checkpoint_inputs,
)


_HEXES = iter("123456789abcdef")


def _digest() -> str:
    return next(_HEXES) * 64


def _ref(
    key: str,
    payload: bytes = b"fixture",
    *,
    version: str,
    encryption: str = "AES256",
    kms_key_arn: str | None = None,
) -> S3ObjectVersion:
    return S3ObjectVersion(
        bucket="${MS_S3_BUCKET}",
        key=key,
        version_id=version,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        server_side_encryption=encryption,
        kms_key_arn=kms_key_arn,
    )


def _fixture():
    public_items = []
    gold_items = []
    predictions = []
    answers = ("yes", "no", "café", "7")
    raw = ("  yes  ", "no extra", "cafe\u0301", "8")
    for position, (family, source_index, answer, prediction) in enumerate(
        zip(("a", "a", "b", "b"), (10, 11, 20, 21), answers, raw, strict=True)
    ):
        item_id = f"fixture/{family}/{source_index}"
        public_items.append(
            {
                "item_id": item_id,
                "max_new_tokens": 8,
                "prompt": f"Question {source_index}\nAnswer:",
                "scorer_id": "memorysplit-independent-exact-v1",
                "source_index": source_index,
                "task": family,
            }
        )
        gold_items.append(
            {
                "canonical_answer": answer,
                "item_id": item_id,
                "oracle_replay": {},
                "source_index": source_index,
                "task": family,
            }
        )
        predictions.append(
            RawPrediction(
                item_id=item_id,
                task=family,
                source_index=source_index,
                raw_prediction=prediction,
                generated_tokens=2,
                stop_reason="eot",
            )
        )
    registry = "a" * 64
    public = {"items": public_items, "registry_sha256": registry}
    sealed = {"items": gold_items, "registry_sha256": registry}
    checkpoint = CheckpointBinding(
        arm="dense",
        seed=0,
        step=15_582,
        sha256="b" * 64,
        object_key=(
            "checkpoints/reasoning-v3/v1/checkpoints/"
            "d135m_dense_reasoning_v3_s0/step0015582.pt"
        ),
        version_id="checkpoint-version",
        bytes=123,
        run_config_path="configs/135m-v3/dense-s0.yaml",
        run_config_sha256="c" * 64,
        initialization_sha256="d" * 64,
        data_order_sha256="e" * 64,
        paired_runtime_sha256="f" * 64,
        paired_corpus_sha256="1" * 64,
        paired_config_sha256="2" * 64,
        checkpoint_kms_key_arn=(
            "arn:aws:kms:us-east-1:${AWS_ACCOUNT_ID}:key/"
            "33333333-3333-4333-8333-333333333333"
        ),
        manifest_key=(
            "evaluations/reasoning-v3/v1/evaluator-only/"
            "manifests/checkpoint-evidence.json"
        ),
        manifest_version_id="manifest-version",
        manifest_sha256="3" * 64,
        evidence_key=(
            "checkpoints/reasoning-v3/v1/evidence/"
            "d135m_dense_reasoning_v3_s0/step0015582/gates.json"
        ),
        evidence_version_id="evidence-version",
        evidence_sha256="4" * 64,
        runtime_lock_key=(
            "checkpoints/reasoning-v3/v1/evidence/runtime/runtime-lock.json"
        ),
        runtime_lock_version_id="runtime-lock-version",
        runtime_lock_object_sha256="5" * 64,
        corpus_receipt_key=(
            "checkpoints/reasoning-v3/v1/evidence/corpus/receipt.json"
        ),
        corpus_receipt_version_id="corpus-version",
        corpus_receipt_object_sha256="6" * 64,
    )
    signer_arn = (
        "arn:aws:kms:us-east-1:${AWS_ACCOUNT_ID}:key/"
        "11111111-1111-4111-8111-111111111111"
    )
    sealed_arn = (
        "arn:aws:kms:us-east-1:${AWS_ACCOUNT_ID}:key/"
        "22222222-2222-4222-8222-222222222222"
    )
    verified = VerifiedAwsAuthority(
        contract_sha256="3" * 64,
        record_sha256="4" * 64,
        record_version_id="authority-version",
        signature_version_id="authority-signature-version",
        signer_key_arn=signer_arn,
        sealed_gold_kms_key_arn=sealed_arn,
        checkpoint_kms_key_arn=(
            "arn:aws:kms:us-east-1:${AWS_ACCOUNT_ID}:key/"
            "33333333-3333-4333-8333-333333333333"
        ),
    )
    activation = _ref(
        "evaluations/reasoning-v3/v1/activation/ACTIVE.json",
        version="activation-version",
    )
    release = ReleaseBinding(
        contract_id="memorysplit-reasoning-v3-eval-v1",
        contract_sha256=verified.contract_sha256,
        authority=verified,
        activation=activation,
        activation_sha256=activation.sha256,
        activation_signature_sha256="5" * 64,
        activation_signing_algorithm="ECDSA_SHA_384",
        model_visible=_ref(
            "evaluations/reasoning-v3/v1/model-visible/quarantine/t/release.json",
            version="model-version",
        ),
        sealed_gold=_ref(
            "evaluations/reasoning-v3/v1/evaluator-only/quarantine/t/gold.json",
            version="gold-version",
            encryption="aws:kms",
            kms_key_arn=sealed_arn,
        ),
        registry_sha256=registry,
        corpus_receipt_sha256="6" * 64,
        runtime_lock_sha256="7" * 64,
    )
    gates = {
        name: GateEvidence(passed=True, evidence_sha256="8" * 64)
        for name in REQUIRED_VALIDITY_GATES
    }
    code = (
        CodeBinding(path="evals/reasoning_v3/runner.py", sha256="9" * 64),
        CodeBinding(path="evals/reasoning_v3/inference.py", sha256="a" * 64),
        CodeBinding(path="evals/reasoning_v3/reporting.py", sha256="b" * 64),
    )
    return public, sealed, predictions, checkpoint, release, gates, code


def _build(**overrides):
    public, sealed, predictions, checkpoint, release, gates, code = _fixture()
    values = {
        "public_release": public,
        "sealed_release": sealed,
        "predictions": predictions,
        "checkpoint": checkpoint,
        "release": release,
        "validity": gates,
        "evaluator_code": code,
        "family_order": ("a", "b"),
        "items_per_family": 2,
        "scorer_id": "memorysplit-independent-exact-v1",
    }
    values.update(overrides)
    return _build_checkpoint_result(**values)


def test_exact_output_preserves_raw_text_and_applies_only_nfc_outer_strip():
    score = _score_exact_output(
        "  cafe\u0301  ",
        "café",
        max_new_tokens=8,
        generated_tokens=2,
        stop_reason="eot",
    )
    assert score.raw_prediction == "  cafe\u0301  "
    assert score.prediction == "café"
    assert score.correct is True
    assert score.valid is True

    extra = _score_exact_output(
        "café extra",
        "café",
        max_new_tokens=8,
        generated_tokens=2,
        stop_reason="eot",
    )
    assert extra.valid is True
    assert extra.correct is False

    over_limit = _score_exact_output(
        "café",
        "café",
        max_new_tokens=8,
        generated_tokens=8,
        stop_reason="max_new_tokens",
    )
    assert over_limit.valid is False
    assert over_limit.correct is False
    assert over_limit.error == "over_limit"

    missing = _score_exact_output(
        None,
        "café",
        max_new_tokens=8,
        generated_tokens=0,
        stop_reason="error",
    )
    assert missing.valid is False
    assert missing.correct is False
    assert missing.error == "missing"


def test_greedy_decoder_uses_argmax_stops_at_eot_and_marks_limit_hits():
    import torch

    class Tokenizer:
        EOT = 9

        @staticmethod
        def encode(_prompt):
            return [1]

        @staticmethod
        def decode(ids):
            return "".join({5: "A", 6: "B", 7: "C"}[token] for token in ids)

    class Model:
        cfg = type("Config", (), {"ctx": 8})()

        def forward_step(self, value, cache):
            batch, width = value.shape
            logits = torch.zeros((batch, width, 10))
            choices = (5, 6) if cache is None else (9, 7)
            for index in range(batch):
                logits[index, -1, choices[index]] = 1
            return logits, object()

    items = [
        {
            "item_id": "a/0",
            "max_new_tokens": 2,
            "prompt": "p0",
            "source_index": 0,
            "task": "a",
        },
        {
            "item_id": "a/1",
            "max_new_tokens": 2,
            "prompt": "p1",
            "source_index": 1,
            "task": "a",
        },
    ]
    predictions = _greedy_decode_model(
        Model(),
        Tokenizer(),
        items,
        device="cpu",
        batch_size=2,
    )
    assert predictions == [
        RawPrediction("a/0", "a", 0, "A", 2, "eot"),
        RawPrediction("a/1", "a", 1, "BC", 2, "max_new_tokens"),
    ]


def test_checkpoint_result_has_closed_rows_and_recomputed_family_macro_scores():
    result = _build()
    assert set(result) == {
        "checkpoint",
        "cohort",
        "evaluator",
        "family_scores",
        "format",
        "items",
        "macro_accuracy",
        "macro_accuracy_fraction",
        "release",
        "run",
        "schema_version",
        "validity",
    }
    assert len(result["items"]) == 4
    assert [row["position"] for row in result["items"]] == [0, 1, 2, 3]
    assert result["items"][0]["raw_prediction"] == "  yes  "
    assert result["items"][2]["prediction"] == "café"
    assert result["release"]["activation_signature_sha256"] == "5" * 64
    assert (
        result["release"]["activation_signing_algorithm"]
        == "ECDSA_SHA_384"
    )
    assert [row["numerator"] for row in result["family_scores"]] == [1, 1]
    assert [row["denominator"] for row in result["family_scores"]] == [2, 2]
    assert result["macro_accuracy"] == pytest.approx(0.5)

    public, sealed, *_ = _fixture()
    _validate_checkpoint_result(
        result,
        public_release=public,
        sealed_release=sealed,
        family_order=("a", "b"),
        items_per_family=2,
        scorer_id="memorysplit-independent-exact-v1",
    )


@pytest.mark.parametrize("mutation", ["correctness", "aggregate", "raw_prediction"])
def test_checkpoint_validation_replays_rows_and_rejects_tampering(mutation: str):
    result = json.loads(canonical_json_bytes(_build()))
    if mutation == "correctness":
        result["items"][0]["correct"] = False
    elif mutation == "aggregate":
        result["family_scores"][0]["numerator"] = 0
    else:
        result["items"][0]["raw_prediction"] = "wrong"

    public, sealed, *_ = _fixture()
    with pytest.raises(RunnerError):
        _validate_checkpoint_result(
            result,
            public_release=public,
            sealed_release=sealed,
            family_order=("a", "b"),
            items_per_family=2,
            scorer_id="memorysplit-independent-exact-v1",
        )


@pytest.mark.parametrize(
    "section",
    ["checkpoint", "cohort", "evaluator", "release", "run", "pairing"],
)
def test_checkpoint_validation_rejects_unknown_nested_schema_fields(section: str):
    result = json.loads(canonical_json_bytes(_build()))
    target = result["run"]["pairing"] if section == "pairing" else result[section]
    target["unexpected"] = True
    public, sealed, *_ = _fixture()
    with pytest.raises(RunnerError, match="fields"):
        _validate_checkpoint_result(
            result,
            public_release=public,
            sealed_release=sealed,
            family_order=("a", "b"),
            items_per_family=2,
            scorer_id="memorysplit-independent-exact-v1",
        )


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate", "identity"])
def test_checkpoint_builder_rejects_missing_extra_duplicate_or_mismatched_rows(mode):
    public, sealed, predictions, checkpoint, release, gates, code = _fixture()
    if mode == "missing":
        predictions = predictions[:-1]
    elif mode == "extra":
        predictions = [*predictions, replace(predictions[-1], item_id="extra")]
    elif mode == "duplicate":
        predictions = [predictions[0], predictions[0], *predictions[2:]]
    else:
        predictions = [replace(predictions[0], source_index=999), *predictions[1:]]
    with pytest.raises(RunnerError):
        _build(
            public_release=public,
            sealed_release=sealed,
            predictions=predictions,
            checkpoint=checkpoint,
            release=release,
            validity=gates,
            evaluator_code=code,
        )


def test_checkpoint_publication_is_evaluator_only_no_replace_and_crash_safe():
    result = _build()
    release = _fixture()[4]

    class FakeAuthority:
        def __init__(self):
            self.objects: dict[str, bytes] = {}
            self.crash = True
            self.last_kwargs = None

        def put_checkpoint_result(self, arm, seed, step, payload, verified):
            key = _checkpoint_result_key(arm, seed, step)
            self.last_kwargs = {
                "arm": arm,
                "seed": seed,
                "step": step,
                "verified": verified,
            }
            if self.crash:
                self.crash = False
                raise RuntimeError("simulated request-body crash")
            prior = self.objects.setdefault(key, payload)
            if prior != payload:
                raise RuntimeError("immutable object differs")
            ref = _ref(
                key,
                payload,
                version="result-version",
                encryption="aws:kms",
                kms_key_arn=verified.sealed_gold_kms_key_arn,
            )
            return authority_module.Task3Publication(
                payload_bytes=payload,
                envelope_bytes=b"signed envelope",
                object_ref=ref,
                created=prior is payload,
            )

    authority = FakeAuthority()
    with pytest.raises(RuntimeError, match="crash"):
        _publish_checkpoint_result(result, release, authority)
    assert authority.objects == {}

    published = _publish_checkpoint_result(result, release, authority)
    repeated = _publish_checkpoint_result(result, release, authority)
    assert repeated.object_ref == published.object_ref
    assert published.object_ref.key == _checkpoint_result_key("dense", 0, 15_582)
    assert "/evaluator-only/results/" in f"/{published.object_ref.key}"
    assert published.object_ref.version_id == "result-version"
    assert authority.last_kwargs == {
        "arm": "dense",
        "seed": 0,
        "step": 15_582,
        "verified": release.authority,
    }

    changed = json.loads(canonical_json_bytes(result))
    changed["macro_accuracy"] = 0.25
    with pytest.raises(RuntimeError, match="differs"):
        _publish_checkpoint_result(changed, release, authority)


def test_checkpoint_bytes_are_verified_by_exact_fixed_bucket_object_version(
    monkeypatch: pytest.MonkeyPatch,
):
    binding = _fixture()[3]
    payload = b"exact checkpoint bytes"
    binding = replace(
        binding,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    class FakeAuthority:
        def __init__(self):
            self.calls = []

        def read_checkpoint(self, ref):
            self.calls.append(ref)
            return payload, _ref(
                ref.key,
                payload,
                version=binding.version_id,
                encryption="aws:kms",
                kms_key_arn=binding.checkpoint_kms_key_arn,
            )

    authority = FakeAuthority()
    monkeypatch.setattr(
        runner_module,
        "_deserialize_checkpoint",
        lambda _payload, *, expected_step: {
            "model": {},
            "model_cfg": runner_module._MODEL_CONFIG,
            "step": expected_step,
        },
    )
    state = runner_module._load_verified_checkpoint(binding, authority)
    assert state["step"] == binding.step
    assert authority.calls[0].key == binding.object_key
    assert authority.calls[0].version_id == binding.version_id

    class WrongVersion(FakeAuthority):
        def read_checkpoint(self, expected):
            body, ref = super().read_checkpoint(expected)
            return body, replace(ref, version_id="wrong-version")

    with pytest.raises(RunnerError, match="exact signed manifest"):
        runner_module._load_verified_checkpoint(binding, WrongVersion())


def test_production_runner_has_no_injectable_authority_contract_gold_or_items():
    assert set(inspect.signature(run_frozen_checkpoint_evaluation).parameters) == {
        "arm",
        "seed",
        "checkpoint_step",
    }
    assert set(inspect.signature(validate_frozen_checkpoint_inputs).parameters) == {
        "arm",
        "seed",
        "checkpoint_step",
    }
    forbidden = {"authority", "contract", "gold", "items", "storage", "verifier"}
    assert not (
        forbidden
        & set(inspect.signature(run_frozen_checkpoint_evaluation).parameters)
    )
    assert not hasattr(runner_module, "score_exact_output")
    assert not hasattr(runner_module, "FrozenRunnerPaths")


def test_checkpoint_deserialization_uses_same_bytes_weights_only_and_blocks_pickle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import torch

    marker = tmp_path / "pickle-executed"

    class Malicious:
        def __reduce__(self):
            return os.system, (f"touch {marker}",)

    with pytest.warns(UserWarning, match="pickle protocol"):
        with pytest.raises(RunnerError, match="safe weights-only"):
            runner_module._deserialize_checkpoint(
                pickle.dumps(Malicious()),
                expected_step=15_582,
            )
    assert not marker.exists()

    buffer = io.BytesIO()
    expected = {"model": {}, "model_cfg": runner_module._MODEL_CONFIG, "step": 15_582}
    torch.save(expected, buffer)
    calls = []
    original = torch.load

    def guarded_load(source, **kwargs):
        calls.append((source, kwargs))
        assert isinstance(source, io.BytesIO)
        assert kwargs["weights_only"] is True
        return original(source, **kwargs)

    monkeypatch.setattr(torch, "load", guarded_load)
    assert runner_module._deserialize_checkpoint(
        buffer.getvalue(),
        expected_step=15_582,
    ) == expected
    assert len(calls) == 1


def test_substituted_checkpoint_bytes_are_rejected_before_deserialization(
    monkeypatch: pytest.MonkeyPatch,
):
    binding = _fixture()[3]
    exact = b"exact signed-manifest checkpoint bytes"
    binding = replace(
        binding,
        bytes=len(exact),
        sha256=hashlib.sha256(exact).hexdigest(),
    )
    called = []

    def deserialize(payload, *, expected_step):
        called.append(payload)
        return {"model": {}, "model_cfg": runner_module._MODEL_CONFIG, "step": expected_step}

    monkeypatch.setattr(runner_module, "_deserialize_checkpoint", deserialize)

    class Substituted:
        @staticmethod
        def read_checkpoint(ref):
            return b"substituted", replace(
                _ref(
                    ref.key,
                    b"substituted",
                    version=ref.version_id,
                    encryption="aws:kms",
                    kms_key_arn=_fixture()[4].authority.checkpoint_kms_key_arn,
                ),
                version_id="changed-version",
            )

    with pytest.raises(RunnerError, match="exact signed manifest"):
        runner_module._load_verified_checkpoint(binding, Substituted())
    assert called == []

    class Exact:
        @staticmethod
        def read_checkpoint(ref):
            return exact, _ref(
                ref.key,
                exact,
                version=ref.version_id,
                encryption="aws:kms",
                kms_key_arn=_fixture()[4].authority.checkpoint_kms_key_arn,
            )

    loaded = runner_module._load_verified_checkpoint(binding, Exact())
    assert loaded["step"] == 15_582
    assert called == [exact]


def _raw_evidence(
    checkpoint: CheckpointBinding,
    release: ReleaseBinding,
) -> dict[str, object]:
    return {
        "arm": checkpoint.arm,
        "evaluator": {"observed_role_arn": EVALUATOR_ROLE_ARN},
        "exact_resume": {
            "resumed_state_sha256": "1" * 64,
            "uninterrupted_state_sha256": "1" * 64,
        },
        "exclusion": {
            "excluded_item_ids": [],
            "expected_item_count": 4,
            "observed_item_count": 4,
        },
        "factual_burden": {
            "observed_fact_tokens": 100,
            "required_min_fact_tokens": 100,
        },
        "format": "memorysplit-reasoning-v3-raw-validity-evidence-v1",
        "identity": {
            "config_sha256": checkpoint.paired_config_sha256,
            "corpus_sha256": checkpoint.paired_corpus_sha256,
            "data_order_sha256": checkpoint.data_order_sha256,
            "initialization_sha256": checkpoint.initialization_sha256,
            "run_config_sha256": checkpoint.run_config_sha256,
            "runtime_sha256": checkpoint.paired_runtime_sha256,
        },
        "imputation": {
            "expected_seeds": list(range(10)),
            "imputed_seeds": [],
            "observed_seeds": list(range(10)),
        },
        "memory_off_leakage": {"leaked_record_count": 0},
        "memory_on_recall": {
            "expected_recalled_records": 100,
            "observed_recalled_records": 100,
        },
        "provider": {
            "expected": runner_module.PROVIDER,
            "observed": runner_module.PROVIDER,
        },
        "registry": {
            "expected_item_count": 4,
            "expected_registry_sha256": release.registry_sha256,
            "observed_item_count": 4,
            "observed_registry_sha256": release.registry_sha256,
        },
        "replacement": {
            "expected_run_id": "d135m_dense_reasoning_v3_s0",
            "observed_run_id": "d135m_dense_reasoning_v3_s0",
            "replacement_run_ids": [],
        },
        "schema_version": 1,
        "seed": checkpoint.seed,
        "step": checkpoint.step,
        "stopping": {
            "completed_steps": list(runner_module.CHECKPOINT_STEPS),
            "continuation_decisions": [
                {"reason": "precommitted", "step": step}
                for step in runner_module.CHECKPOINT_STEPS
            ],
            "planned_steps": list(runner_module.CHECKPOINT_STEPS),
        },
        "substitution": {
            "loaded_checkpoint_sha256": checkpoint.sha256,
            "manifest_checkpoint_sha256": checkpoint.sha256,
        },
    }


def test_raw_evidence_derives_gates_and_rejects_favorable_boolean_fabrication():
    checkpoint = _fixture()[3]
    release = _fixture()[4]
    context = runner_module.EvidenceContext(
        checkpoint=checkpoint,
        expected_item_count=4,
        manifest_complete=True,
        registry_sha256=release.registry_sha256,
    )
    derived = runner_module._derive_validity_from_evidence(
        _raw_evidence(checkpoint, release),
        context,
    )
    assert set(derived) == set(REQUIRED_VALIDITY_GATES)
    assert all(gate.passed for gate in derived.values())

    fabricated = _raw_evidence(checkpoint, release)
    fabricated["passed"] = True
    with pytest.raises(RunnerError, match="fields"):
        runner_module._derive_validity_from_evidence(fabricated, context)

    leaked = _raw_evidence(checkpoint, release)
    leaked["memory_off_leakage"]["leaked_record_count"] = 1
    assert not runner_module._derive_validity_from_evidence(
        leaked,
        context,
    )["memory_off_leakage"].passed
    leaked["memory_off_leakage"]["passed"] = True
    with pytest.raises(RunnerError, match="fields"):
        runner_module._derive_validity_from_evidence(leaked, context)


def test_actual_runtime_lock_rejects_alternate_root_lookalikes(tmp_path: Path):
    lock = runner_module._actual_runtime_identity()
    lookalike = tmp_path / "train" / "model.py"
    lookalike.parent.mkdir()
    lookalike.write_text("# attacker-controlled lookalike\n", encoding="utf-8")
    changed = json.loads(canonical_json_bytes(lock))
    model = next(
        row for row in changed["modules"] if row["module"] == "train.model"
    )
    model["sha256"] = hashlib.sha256(lookalike.read_bytes()).hexdigest()
    with pytest.raises(RunnerError, match="actual imported runtime"):
        runner_module._verify_runtime_lock(changed)


def test_runtime_lock_binds_actual_serializer_and_effective_tokenizer_data():
    lock = runner_module._actual_runtime_identity()
    modules = {row["module"]: row for row in lock["modules"]}
    assert modules["corpusgen.parallel.canonical"]["path"] == (
        "corpusgen/parallel/canonical.py"
    )
    assert lock["tokenizer"]["effective_sha256"]
    assert lock["tokenizer"]["cache_root"] == ".tiktoken_cache"
    assert {row["name"] for row in lock["tokenizer"]["cache_files"]} == {
        "6c7ea1a7e38e3a7f062df639a5b80947f075ffe6",
        "6d1cbeee0f20b3d9449abfede4726ed8212e3aee",
    }

    for mutation in ("serializer", "tokenizer_source", "effective", "cache"):
        changed = json.loads(canonical_json_bytes(lock))
        if mutation == "serializer":
            modules = {
                row["module"]: row for row in changed["modules"]
            }
            modules["corpusgen.parallel.canonical"]["sha256"] = "0" * 64
        elif mutation == "tokenizer_source":
            modules = {
                row["module"]: row for row in changed["modules"]
            }
            modules["train.tokenizer"]["sha256"] = "0" * 64
        elif mutation == "effective":
            changed["tokenizer"]["effective_sha256"] = "0" * 64
        else:
            changed["tokenizer"]["cache_files"][0]["sha256"] = "0" * 64
        with pytest.raises(RunnerError, match="signed runtime lock"):
            runner_module._verify_runtime_lock(changed)


def test_runtime_admission_rejects_actual_serializer_tokenizer_and_cache_drift(
    monkeypatch: pytest.MonkeyPatch,
):
    lock = runner_module._actual_runtime_identity()
    canonical_path = (
        runner_module.ROOT / "corpusgen/parallel/canonical.py"
    ).resolve()
    cache_path = (
        runner_module.ROOT
        / lock["tokenizer"]["cache_root"]
        / lock["tokenizer"]["cache_files"][0]["name"]
    ).resolve()
    real_read_bytes = Path.read_bytes

    with monkeypatch.context() as patch:
        patch.setattr(
            Path,
            "read_bytes",
            lambda path: (
                real_read_bytes(path) + b"\n# serializer drift\n"
                if path.resolve() == canonical_path
                else real_read_bytes(path)
            ),
        )
        with pytest.raises(RunnerError, match="signed runtime lock"):
            runner_module._verify_runtime_lock(lock)

    expansion = importlib.import_module("corpusgen.reasoning_expansion")
    with monkeypatch.context() as patch:
        patch.setattr(
            expansion,
            "effective_tokenizer_sha256",
            lambda: "0" * 64,
        )
        with pytest.raises(RunnerError, match="signed runtime lock"):
            runner_module._verify_runtime_lock(lock)

    with monkeypatch.context() as patch:
        patch.setattr(
            Path,
            "read_bytes",
            lambda path: (
                real_read_bytes(path) + b"\ncache drift\n"
                if path.resolve() == cache_path
                else real_read_bytes(path)
            ),
        )
        with pytest.raises(
            RunnerError,
            match="signed runtime lock|frozen tokenizer cache file",
        ):
            runner_module._verify_runtime_lock(lock)


def test_closed_checkpoint_replay_covers_all_7168_rows_with_exact_counts():
    public_items = []
    sealed_items = []
    predictions = []
    for family_index, family in enumerate(FAMILY_ORDER):
        for item in range(512):
            item_id = f"{family}/{item:03d}"
            answer = str((family_index + item) % 3)
            public_items.append(
                {
                    "item_id": item_id,
                    "max_new_tokens": 8,
                    "prompt": f"Question {item}\nAnswer:",
                    "scorer_id": "memorysplit-independent-exact-v1",
                    "source_index": 2_000_000_000 + family_index * 10_000 + item,
                    "task": family,
                }
            )
            sealed_items.append(
                {
                    "canonical_answer": answer,
                    "item_id": item_id,
                    "oracle_replay": {},
                    "source_index": 2_000_000_000 + family_index * 10_000 + item,
                    "task": family,
                }
            )
            predictions.append(
                RawPrediction(
                    item_id=item_id,
                    task=family,
                    source_index=2_000_000_000 + family_index * 10_000 + item,
                    raw_prediction=answer if item % 2 == 0 else "wrong",
                    generated_tokens=1,
                    stop_reason="eot",
                )
            )
    public = {"items": public_items, "registry_sha256": "a" * 64}
    sealed = {"items": sealed_items, "registry_sha256": "a" * 64}
    _, _, _, checkpoint, release, gates, code = _fixture()
    result = _build_checkpoint_result(
        public_release=public,
        sealed_release=sealed,
        predictions=predictions,
        checkpoint=checkpoint,
        release=replace(release, registry_sha256="a" * 64),
        validity=gates,
        evaluator_code=code,
        family_order=FAMILY_ORDER,
        items_per_family=512,
        scorer_id="memorysplit-independent-exact-v1",
    )
    assert len(result["items"]) == 7_168
    assert all(row["numerator"] == 256 for row in result["family_scores"])
    assert result["macro_accuracy_fraction"] == {
        "denominator": 2,
        "numerator": 1,
    }


def test_task3_boundary_fixes_exact_keys_operations_kms_and_no_delete():
    boundary = authority_module._parse_aws_boundary_record(
        authority_module.AWS_BOUNDARY_CONFIG_PATH.read_bytes()
    )
    task3 = boundary["task3"]
    assert task3["storage"] == {
        "checkpoint_storage_prefix": "checkpoints/reasoning-v3/v1/",
        "checkpoint_manifest_key": (
            "evaluations/reasoning-v3/v1/evaluator-only/"
            "manifests/checkpoint-evidence.json"
        ),
        "checkpoint_prefix": "checkpoints/reasoning-v3/v1/checkpoints/",
        "evidence_prefix": "checkpoints/reasoning-v3/v1/evidence/",
        "report_key": (
            "evaluations/reasoning-v3/v1/evaluator-only/"
            "reports/frozen-scientific-inference.json"
        ),
        "result_prefix": (
            "evaluations/reasoning-v3/v1/evaluator-only/results/"
        ),
    }
    assert task3["kms"] == {
        "checkpoint_encryption": "aws:kms",
        "checkpoint_key_alias": authority_module.CHECKPOINT_KMS_KEY_ALIAS,
        "checkpoint_key_spec": "SYMMETRIC_DEFAULT",
        "evaluator_artifact_encryption": "aws:kms",
        "evaluator_artifact_key_alias": authority_module.SEALED_GOLD_KMS_KEY_ALIAS,
        "evaluator_artifact_key_spec": "SYMMETRIC_DEFAULT",
        "signing_algorithm": "ECDSA_SHA_384",
        "signing_key_alias": authority_module.SIGNER_KEY_ALIAS,
    }
    assert task3["operations"] == {
        "checkpoint_reader": [
            "kms:Decrypt",
            "kms:DescribeKey",
            "s3:GetObjectVersion",
        ],
        "checkpoint_writer": [
            "kms:DescribeKey",
            "kms:GenerateDataKey",
            "s3:PutObject",
        ],
        "evidence_reader": [
            "kms:Decrypt",
            "kms:DescribeKey",
            "s3:GetObjectVersion",
        ],
        "evidence_writer": [
            "kms:DescribeKey",
            "kms:GenerateDataKey",
            "s3:PutObject",
        ],
        "manifest": ["kms:Sign", "kms:Verify", "s3:GetObjectVersion", "s3:PutObject"],
        "report": ["kms:Sign", "kms:Verify", "s3:GetObjectVersion", "s3:PutObject"],
        "result": ["kms:Sign", "kms:Verify", "s3:GetObjectVersion", "s3:PutObject"],
    }
    assert all(
        "Delete" not in operation
        for operations in task3["operations"].values()
        for operation in operations
    )
    assert task3["required_controls"]["no_overwrite"] is True
    assert task3["required_controls"]["exact_version_reads"] is True


def test_trainer_checkpoint_boundary_is_separate_and_cannot_access_evaluator_gold():
    boundary = authority_module._parse_aws_boundary_record(
        authority_module.AWS_BOUNDARY_CONFIG_PATH.read_bytes()
    )
    task3 = boundary["task3"]
    assert authority_module.TRAINER_ROLE_ARN == (
        "arn:aws:iam::${AWS_ACCOUNT_ID}:role/memorysplit-reasoning-v3-trainer"
    )
    assert authority_module.CHECKPOINT_KMS_KEY_ALIAS == (
        "alias/memorysplit-reasoning-v3-checkpoints-v1"
    )
    assert authority_module.CHECKPOINT_STORAGE_PREFIX == (
        "checkpoints/reasoning-v3/v1/"
    )
    assert task3["roles"] == {
        "checkpoint_reader_role_arn": authority_module.EVALUATOR_ROLE_ARN,
        "checkpoint_writer_role_arn": authority_module.TRAINER_ROLE_ARN,
        "evaluator_artifact_writer_role_arn": authority_module.EVALUATOR_ROLE_ARN,
    }
    assert task3["kms"]["checkpoint_key_alias"] == (
        authority_module.CHECKPOINT_KMS_KEY_ALIAS
    )
    assert task3["kms"]["checkpoint_key_spec"] == "SYMMETRIC_DEFAULT"
    assert task3["kms"]["evaluator_artifact_key_alias"] == (
        authority_module.SEALED_GOLD_KMS_KEY_ALIAS
    )
    assert task3["storage"]["checkpoint_storage_prefix"] == (
        authority_module.CHECKPOINT_STORAGE_PREFIX
    )
    assert task3["storage"]["checkpoint_prefix"].startswith(
        authority_module.CHECKPOINT_STORAGE_PREFIX
    )
    assert task3["storage"]["evidence_prefix"].startswith(
        authority_module.CHECKPOINT_STORAGE_PREFIX
    )
    assert task3["operations"]["checkpoint_writer"] == [
        "kms:DescribeKey",
        "kms:GenerateDataKey",
        "s3:PutObject",
    ]
    assert task3["operations"]["checkpoint_reader"] == [
        "kms:Decrypt",
        "kms:DescribeKey",
        "s3:GetObjectVersion",
    ]
    assert task3["required_controls"]["trainer_denied_evaluator_only_prefix"] is True
    assert task3["required_controls"]["trainer_denied_sealed_gold_kms"] is True
    assert task3["required_controls"]["trainer_denied_signer_kms"] is True
    assert authority_module.CHECKPOINT_KMS_KEY_ALIAS != (
        authority_module.SEALED_GOLD_KMS_KEY_ALIAS
    )


def test_trainer_authority_writes_only_immutable_checkpoint_boundary_objects():
    checkpoint_arn = (
        "arn:aws:kms:us-east-1:${AWS_ACCOUNT_ID}:key/"
        "33333333-3333-4333-8333-333333333333"
    )

    class Sts:
        @staticmethod
        def get_caller_identity():
            return {
                "Account": "${AWS_ACCOUNT_ID}",
                "Arn": (
                    "arn:aws:sts::${AWS_ACCOUNT_ID}:assumed-role/"
                    "memorysplit-reasoning-v3-trainer/test"
                ),
            }

    class Kms:
        @staticmethod
        def describe_key(**kwargs):
            assert kwargs["KeyId"] == authority_module.CHECKPOINT_KMS_KEY_ALIAS
            return {
                "KeyMetadata": {
                    "Arn": checkpoint_arn,
                    "Enabled": True,
                    "KeySpec": "SYMMETRIC_DEFAULT",
                    "KeyState": "Enabled",
                    "KeyUsage": "ENCRYPT_DECRYPT",
                }
            }

    class S3:
        def __init__(self):
            self.put_calls = []

        def put_object(self, **kwargs):
            self.put_calls.append(kwargs)
            return {
                "SSEKMSKeyId": checkpoint_arn,
                "ServerSideEncryption": "aws:kms",
                "VersionId": f"version-{len(self.put_calls)}",
            }

    trainer = object.__new__(authority_module._FixedAwsTrainerAuthority)
    trainer._sts = Sts()
    trainer._kms = Kms()
    trainer._s3 = S3()
    checkpoint = trainer.put_checkpoint("dense", 0, 15_582, b"weights")
    evidence = trainer.put_gate_evidence(
        "dense",
        0,
        15_582,
        canonical_json_bytes({"raw": "evidence"}),
    )
    assert checkpoint.key == (
        "checkpoints/reasoning-v3/v1/checkpoints/"
        "d135m_dense_reasoning_v3_s0/step0015582.pt"
    )
    assert evidence.key == (
        "checkpoints/reasoning-v3/v1/evidence/"
        "d135m_dense_reasoning_v3_s0/step0015582/gates.json"
    )
    assert all(call["IfNoneMatch"] == "*" for call in trainer._s3.put_calls)
    assert all(
        call["SSEKMSKeyId"] == checkpoint_arn
        and call["ServerSideEncryption"] == "aws:kms"
        for call in trainer._s3.put_calls
    )
    assert not hasattr(trainer, "put_sealed_gold")
    assert not hasattr(trainer, "read_sealed_gold")
    assert not hasattr(trainer, "sign")
    with pytest.raises(authority_module.AwsAuthorityError, match="operation"):
        trainer._trainer_key("sealed_gold", "dense", 0, 15_582)


@pytest.mark.parametrize(
    ("encryption", "kms_arn"),
    [
        (
            "AES256",
            "arn:aws:kms:us-east-1:${AWS_ACCOUNT_ID}:key/"
            "33333333-3333-4333-8333-333333333333",
        ),
        (
            "aws:kms",
            "arn:aws:kms:us-east-1:${AWS_ACCOUNT_ID}:key/"
            "99999999-9999-4999-8999-999999999999",
        ),
    ],
)
def test_checkpoint_reader_rejects_wrong_sse_mode_or_checkpoint_kms(
    encryption: str,
    kms_arn: str,
):
    verified = _fixture()[4].authority
    key = (
        "checkpoints/reasoning-v3/v1/checkpoints/"
        "d135m_dense_reasoning_v3_s0/step0015582.pt"
    )
    wrong = _ref(
        key,
        version="version",
        encryption=encryption,
        kms_key_arn=kms_arn,
    )

    class Sts:
        @staticmethod
        def get_caller_identity():
            return {
                "Account": "${AWS_ACCOUNT_ID}",
                "Arn": (
                    "arn:aws:sts::${AWS_ACCOUNT_ID}:assumed-role/"
                    "memorysplit-reasoning-v3-evaluator/test"
                ),
            }

    class NeverCalled:
        def __getattr__(self, _name):
            raise AssertionError("invalid encryption reached S3")

    adapter = object.__new__(authority_module._FixedAwsEvaluatorAuthority)
    adapter._sts = Sts()
    adapter._s3 = NeverCalled()
    with pytest.raises(authority_module.AwsAuthorityError, match="checkpoint"):
        adapter.read_checkpoint(wrong, verified)


def test_signed_result_lost_response_retry_compares_verified_unsigned_payload():
    payload = canonical_json_bytes({"result": "same unsigned scientific payload"})
    verified = _fixture()[4].authority

    class Sts:
        @staticmethod
        def get_caller_identity():
            return {
                "Account": "${AWS_ACCOUNT_ID}",
                "Arn": (
                    "arn:aws:sts::${AWS_ACCOUNT_ID}:assumed-role/"
                    "memorysplit-reasoning-v3-evaluator/test"
                ),
            }

    class PreconditionFailed(Exception):
        response = {"Error": {"Code": "PreconditionFailed"}}

    class Kms:
        def __init__(self):
            self.messages = {}
            self.counter = 0

        def sign(self, **kwargs):
            self.counter += 1
            signature = f"signature-{self.counter}".encode()
            self.messages[signature] = kwargs["Message"]
            return {
                "KeyId": verified.signer_key_arn,
                "Signature": signature,
                "SigningAlgorithm": "ECDSA_SHA_384",
            }

        def verify(self, **kwargs):
            return {
                "KeyId": verified.signer_key_arn,
                "SignatureValid": self.messages.get(kwargs["Signature"])
                == kwargs["Message"],
                "SigningAlgorithm": "ECDSA_SHA_384",
            }

    class S3:
        def __init__(self):
            self.body = None
            self.metadata = None
            self.put_calls = []

        def put_object(self, **kwargs):
            self.put_calls.append(kwargs)
            if self.body is None:
                self.body = kwargs["Body"]
                self.metadata = kwargs["Metadata"]
                raise RuntimeError("lost successful response")
            raise PreconditionFailed()

        def get_object(self, **_kwargs):
            return {
                "Body": io.BytesIO(self.body),
                "Metadata": self.metadata,
                "SSEKMSKeyId": verified.sealed_gold_kms_key_arn,
                "ServerSideEncryption": "aws:kms",
                "VersionId": "immutable-result-version",
            }

    adapter = object.__new__(authority_module._FixedAwsEvaluatorAuthority)
    adapter._sts = Sts()
    adapter._kms = Kms()
    adapter._s3 = S3()
    first = adapter.put_checkpoint_result(
        "dense",
        0,
        15_582,
        payload,
        verified,
    )
    second = adapter.put_checkpoint_result(
        "dense",
        0,
        15_582,
        payload,
        verified,
    )
    assert first.object_ref == second.object_ref
    assert first.envelope_bytes == second.envelope_bytes
    assert first.payload_bytes == second.payload_bytes == payload
    assert adapter._kms.counter == 2
    assert adapter._s3.put_calls[0]["IfNoneMatch"] == "*"
    assert adapter._s3.put_calls[0]["ServerSideEncryption"] == "aws:kms"
    assert (
        adapter._s3.put_calls[0]["SSEKMSKeyId"]
        == verified.sealed_gold_kms_key_arn
    )


def test_task3_authority_rejects_wrong_key_pattern_operation_and_encryption():
    verified = _fixture()[4].authority

    class Sts:
        @staticmethod
        def get_caller_identity():
            return {
                "Account": "${AWS_ACCOUNT_ID}",
                "Arn": (
                    "arn:aws:sts::${AWS_ACCOUNT_ID}:assumed-role/"
                    "memorysplit-reasoning-v3-evaluator/test"
                ),
            }

    class NeverCalled:
        def __getattr__(self, _name):
            raise AssertionError("rejected request reached AWS")

    adapter = object.__new__(authority_module._FixedAwsEvaluatorAuthority)
    adapter._sts = Sts()
    adapter._kms = NeverCalled()
    adapter._s3 = NeverCalled()
    wrong = _ref(
        "attacker/checkpoint.pt",
        version="version",
        encryption="aws:kms",
        kms_key_arn=verified.sealed_gold_kms_key_arn,
    )
    with pytest.raises(authority_module.AwsAuthorityError, match="checkpoint"):
        adapter.read_checkpoint(wrong, verified)
    with pytest.raises(authority_module.AwsAuthorityError, match="matrix"):
        adapter.put_checkpoint_result("other", 0, 15_582, b"{}", verified)
    with pytest.raises(authority_module.AwsAuthorityError, match="operation"):
        adapter._task3_key("delete", "dense", 0, 15_582)


def test_forged_canonical_local_sidecar_cannot_replace_signed_manifest():
    source = Path(runner_module.__file__).read_text(encoding="utf-8")
    assert ".authority.json" not in source
    assert "_load_checkpoint_inputs" not in source
    assert hasattr(authority_module._FixedAwsEvaluatorAuthority, "read_checkpoint_manifest")


def _manifest_ref_dict(ref: S3ObjectVersion) -> dict[str, object]:
    value = {
        "bucket": ref.bucket,
        "bytes": ref.bytes,
        "key": ref.key,
        "server_side_encryption": ref.server_side_encryption,
        "sha256": ref.sha256,
        "version_id": ref.version_id,
    }
    if ref.kms_key_arn is not None:
        value["kms_key_arn"] = ref.kms_key_arn
    return value


def _task3_manifest_fixture():
    release = _fixture()[4]
    kms = release.authority.checkpoint_kms_key_arn

    def evidence_ref(key: str, digest: str) -> S3ObjectVersion:
        return S3ObjectVersion(
            bucket=authority_module.STORAGE_BUCKET,
            key=key,
            version_id=f"version-{digest[:8]}",
            bytes=100,
            sha256=digest,
            server_side_encryption="aws:kms",
            kms_key_arn=kms,
        )

    runtime = evidence_ref(
        f"{authority_module.TASK3_EVIDENCE_PREFIX}runtime/runtime-lock.json",
        "1" * 64,
    )
    corpus = evidence_ref(
        f"{authority_module.TASK3_EVIDENCE_PREFIX}corpus/receipt.json",
        "2" * 64,
    )
    cells = []
    for seed in range(10):
        for arm in ("dense", "split90"):
            run = f"d135m_{arm}_reasoning_v3_s{seed}"
            pairing = {
                kind: _manifest_ref_dict(
                    evidence_ref(
                        (
                            f"{authority_module.TASK3_EVIDENCE_PREFIX}"
                            f"{run}/pairing/{kind}.json"
                        ),
                        format(
                            (seed * 7 + len(kind) + (0 if arm == "dense" else 1))
                            % 15
                            + 1,
                            "x",
                        )
                        * 64,
                    )
                )
                for kind in (
                    "config",
                    "corpus",
                    "data_order",
                    "initialization",
                    "runtime",
                )
            }
            for step in runner_module.CHECKPOINT_STEPS:
                checkpoint = evidence_ref(
                    (
                        f"{authority_module.TASK3_CHECKPOINT_PREFIX}"
                        f"{run}/step{step:07d}.pt"
                    ),
                    format((seed + step + (arm == "split90")) % 15 + 1, "x") * 64,
                )
                config = evidence_ref(
                    (
                        f"{authority_module.TASK3_EVIDENCE_PREFIX}"
                        f"{run}/config.json"
                    ),
                    format(seed + 1, "x") * 64,
                )
                evidence = evidence_ref(
                    (
                        f"{authority_module.TASK3_EVIDENCE_PREFIX}"
                        f"{run}/step{step:07d}/gates.json"
                    ),
                    format((seed + step) % 15 + 1, "x") * 64,
                )
                cells.append(
                    {
                        "arm": arm,
                        "checkpoint": _manifest_ref_dict(checkpoint),
                        "evidence": _manifest_ref_dict(evidence),
                        "pairing": pairing,
                        "run_config": _manifest_ref_dict(config),
                        "seed": seed,
                        "step": step,
                    }
                )
    payload = {
        "activation": _manifest_ref_dict(release.activation),
        "authority_record_sha256": release.authority.record_sha256,
        "authority_record_version_id": release.authority.record_version_id,
        "cells": cells,
        "cohort_id": "memorysplit-exploratory-v3-135m-aws-n10",
        "contract_sha256": release.contract_sha256,
        "corpus_receipt": _manifest_ref_dict(corpus),
        "format": "memorysplit-reasoning-v3-checkpoint-evidence-manifest-v1",
        "runtime_lock": _manifest_ref_dict(runtime),
        "schema_version": 1,
    }
    manifest_bytes = canonical_json_bytes(payload)
    manifest_ref = S3ObjectVersion(
        bucket=authority_module.STORAGE_BUCKET,
        key=(
            f"{authority_module.STORAGE_PREFIX}/evaluator-only/"
            "manifests/checkpoint-evidence.json"
        ),
        version_id="signed-manifest-version",
        bytes=len(manifest_bytes),
        sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        server_side_encryption="aws:kms",
        kms_key_arn=release.authority.sealed_gold_kms_key_arn,
    )
    return payload, manifest_ref, release


def test_checkpoint_manifest_is_closed_complete_and_binds_every_exact_version():
    payload, manifest_ref, release = _task3_manifest_fixture()
    parsed = runner_module._parse_checkpoint_manifest(
        payload,
        manifest_ref,
        release,
    )
    assert len(parsed.cells) == 100
    assert parsed.ref == manifest_ref
    assert parsed.cells[("dense", 0, 15_582)].checkpoint.version_id

    incomplete = json.loads(canonical_json_bytes(payload))
    incomplete["cells"].pop()
    with pytest.raises(RunnerError, match="100"):
        runner_module._parse_checkpoint_manifest(
            incomplete,
            manifest_ref,
            release,
        )

    unknown = json.loads(canonical_json_bytes(payload))
    unknown["cells"][0]["favorable_passed"] = True
    with pytest.raises(RunnerError, match="fields"):
        runner_module._parse_checkpoint_manifest(
            unknown,
            manifest_ref,
            release,
        )


def test_canonical_manifest_with_forged_kms_signature_is_rejected():
    _, _, release = _task3_manifest_fixture()
    unsigned = canonical_json_bytes({"manifest": "forged"})
    signature_document = authority_module._task3_signature_document_bytes(
        b"forged-signature",
        unsigned,
        release.authority.signer_key_arn,
    )
    envelope = canonical_json_bytes(
        {
            "payload": json.loads(unsigned),
            "signature": json.loads(signature_document),
        }
    )

    class Sts:
        @staticmethod
        def get_caller_identity():
            return {
                "Account": "${AWS_ACCOUNT_ID}",
                "Arn": (
                    "arn:aws:sts::${AWS_ACCOUNT_ID}:assumed-role/"
                    "memorysplit-reasoning-v3-evaluator/test"
                ),
            }

    class Kms:
        @staticmethod
        def verify(**_kwargs):
            return {
                "KeyId": release.authority.signer_key_arn,
                "SignatureValid": False,
                "SigningAlgorithm": "ECDSA_SHA_384",
            }

    class S3:
        @staticmethod
        def get_object(**_kwargs):
            return {
                "Body": io.BytesIO(envelope),
                "Metadata": {"sha256": hashlib.sha256(envelope).hexdigest()},
                "SSEKMSKeyId": release.authority.sealed_gold_kms_key_arn,
                "ServerSideEncryption": "aws:kms",
                "VersionId": "forged-manifest-version",
            }

    adapter = object.__new__(authority_module._FixedAwsEvaluatorAuthority)
    adapter._sts = Sts()
    adapter._kms = Kms()
    adapter._s3 = S3()
    with pytest.raises(authority_module.AwsAuthorityError, match="signature"):
        adapter.read_checkpoint_manifest(release.authority)
