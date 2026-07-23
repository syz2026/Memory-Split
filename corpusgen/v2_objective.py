"""Pinned objective-generator worker for MemorySplit v2.

The production materializer starts one isolated worker per upstream generator.
Isolation is intentional: RuleTaker and ProntoQA both use top-level module
names, while CLRS and Reasoning Gym import large optional dependency trees.
Every worker receives a record ordinal and reseeds the native implementation,
so restarting at an arbitrary ordinal produces the same question and answer.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Any

WORKER_FORMAT = "memorysplit-v2-objective-worker-v1"
PROCEDURAL_PROVIDERS = (
    "deepmind_mathematics_generator",
    "clrs_text",
    "ruletaker",
    "prontoqa",
    "reasoning_gym_exact_answer",
)
_SEED_ATTEMPTS = 128
_SEED_MULTIPLIER = 0x9E3779B1
_STDLIB_RANDINT = random.randint


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _seed(provider: str, index: int, attempt: int = 0) -> int:
    if index < 0 or index >= (1 << 32) // _SEED_ATTEMPTS:
        raise ValueError(
            "objective record index exceeds the collision-free seed domain"
        )
    if attempt < 0 or attempt >= _SEED_ATTEMPTS:
        raise ValueError("objective generation attempt exceeds the frozen seed domain")
    # Native generators accept only uint32 seeds.  A truncated digest would
    # collide at birthday-bound scale during a multi-million-record build.
    # This odd affine permutation is injective over every (index, attempt)
    # pair admitted above, while the provider digest separates streams.
    ordinal = index * _SEED_ATTEMPTS + attempt
    offset = int.from_bytes(hashlib.sha256(provider.encode()).digest()[:4], "big")
    return (_SEED_MULTIPLIER * ordinal + offset) & 0xFFFFFFFF


def _runtime_versions(distributions: tuple[str, ...]) -> dict[str, str]:
    versions = {"python": ".".join(map(str, sys.version_info[:3]))}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "stdlib-or-unpackaged"
    return dict(sorted(versions.items()))


def _flatten_modules(value: dict[str, Any], prefix: str = ""):
    for name in sorted(value):
        item = value[name]
        qualified = f"{prefix}__{name}" if prefix else name
        if isinstance(item, dict):
            yield from _flatten_modules(item, qualified)
        else:
            yield qualified, item


def _deepmind_expand_entities(composition, context, **kwargs):
    """Replay upstream entity expansion without object-hash ordering."""

    expanded = kwargs.copy()
    entities = []
    seen_identities = set()

    def append_once(entity):
        identity = id(entity)
        if identity not in seen_identities:
            seen_identities.add(identity)
            entities.append(entity)

    # Upstream constructs ``set(context.child_entities)`` here. ``Entity``
    # retains object-identity hashing, so allocator history changes the list
    # presented to the subsequent seeded shuffle. Preserve the native
    # insertion order and native shuffle while deduplicating by the same object
    # identity. This makes direct ordinal replay independent of earlier calls.
    for entity in context.child_entities:
        append_once(entity)
    for key, maybe_entity in kwargs.items():
        if isinstance(maybe_entity, composition.Entity):
            append_once(maybe_entity)
            expanded[key] = maybe_entity.handle
    random.shuffle(entities)

    child_descriptions = []
    for entity in entities:
        child_descriptions.append(entity.child_description)
        if not entity.expression_used:
            child_descriptions.append(entity.description)
    child_description = " ".join(value for value in child_descriptions if value)
    return child_description, expanded


def _deepmind_randint(lower, upper):
    """Preserve the integral-float bounds accepted by older Python runtimes."""

    if isinstance(lower, float) and lower.is_integer():
        lower = int(lower)
    if isinstance(upper, float) and upper.is_integer():
        upper = int(upper)
    return _STDLIB_RANDINT(lower, upper)


class _DeepMindMathematics:
    distributions = ("absl-py", "numpy", "six", "sympy")

    def __init__(self, root: Path):
        import numpy as np
        from sympy.core import random as sympy_random

        if int(np.__version__.split(".", 1)[0]) >= 2:
            raise RuntimeError(
                "the pinned DeepMind mathematics source requires NumPy <2 "
                "(it uses ndarray.itemset)"
            )
        # Upstream constructs the train-module registry during initialization,
        # before ``generate`` gets an ordinal-specific seed.  Seed that
        # construction too; otherwise fresh worker processes can capture
        # different module state even though every later record call reseeds.
        initialization_seed = _seed("deepmind_mathematics_generator:init", 0)
        random.seed(initialization_seed)
        np.random.seed(initialization_seed)
        sympy_random.seed(initialization_seed)
        self._sympy_random = sympy_random
        sys.path.insert(0, str(root))
        from mathematics_dataset.util import composition

        composition.expand_entities = lambda context, **kwargs: (
            _deepmind_expand_entities(composition, context, **kwargs)
        )
        from mathematics_dataset.modules import modules

        def full_entropy(bounds):
            return bounds

        self._modules = tuple(_flatten_modules(modules.train(full_entropy)))
        if not self._modules:
            raise RuntimeError("DeepMind mathematics exposed no train modules")

    def generate(self, index: int) -> dict[str, Any]:
        import numpy as np

        module_name, module = self._modules[index % len(self._modules)]
        for attempt in range(128):
            seed = _seed("deepmind_mathematics_generator", index, attempt)
            random.seed(seed)
            np.random.seed(seed)
            self._sympy_random.seed(seed)
            native_randint = random.randint
            random.randint = _deepmind_randint
            try:
                problem = module()
            finally:
                random.randint = native_randint
            question = str(problem.question)
            answer = str(problem.answer)
            if question and answer and len(question) <= 160 and len(answer) <= 30:
                return {
                    "answer": answer,
                    "metadata": {
                        "attempt": attempt,
                        "module": module_name,
                        "seed": seed,
                    },
                    "question": question,
                }
        raise RuntimeError(
            f"DeepMind mathematics record {index} exceeded 128 valid-sample attempts"
        )


class _ClrsText:
    distributions = (
        "absl-py",
        "attrs",
        "chex",
        "jax",
        "jaxlib",
        "numpy",
        "tensorflow",
    )
    _LENGTHS = (4, 8, 16)

    def __init__(self, root: Path):
        sys.path.insert(0, str(root))
        # Import only the pinned native CLRS-Text generator and sampler tree.
        # The repository's top-level ``clrs`` package eagerly imports training
        # models and TensorFlow, neither of which participates in text
        # generation.  A normal namespace package avoids those unrelated
        # runtime dependencies without replacing any generator implementation.
        package = types.ModuleType("clrs")
        package.__path__ = [str(root / "clrs")]
        package.__package__ = "clrs"
        sys.modules["clrs"] = package
        from clrs._src.clrs_text.huggingface_generators import clrs_generator
        from clrs._src.specs import CLRS_30_ALGS_SETTINGS

        self._generator = clrs_generator
        self._algorithms = tuple(sorted(CLRS_30_ALGS_SETTINGS))
        if not self._algorithms:
            raise RuntimeError("CLRS exposed no algorithms")

    def generate(self, index: int) -> dict[str, Any]:
        algorithm = self._algorithms[index % len(self._algorithms)]
        length = self._LENGTHS[(index // len(self._algorithms)) % len(self._LENGTHS)]
        seed = _seed("clrs_text", index)
        row = next(
            self._generator(
                {algorithm: [length]},
                num_samples=1,
                use_hints=True,
                seed=seed,
            )
        )
        return {
            "answer": str(row["answer"]),
            "metadata": {
                "algorithm": algorithm,
                "length": length,
                "seed": seed,
                "use_hints": True,
            },
            "question": str(row["question"]),
        }


class _RuleTaker:
    distributions = ("nltk", "numpy", "problog", "PySDD", "tqdm")

    def __init__(self, root: Path):
        sys.path.insert(0, str(root))
        import nltk
        import theory_generator

        self._module = theory_generator
        grammar_path = (
            root / "grammars_and_config" / "grammars" / "ruletaker_grammar_theory1.txt"
        )
        config_path = (
            root
            / "grammars_and_config"
            / "config"
            / "ruletaker_theory_generator_config_theory1.json"
        )
        config = json.loads(config_path.read_bytes())
        with grammar_path.open(encoding="utf-8") as handle:
            productions = theory_generator.preprocess_pcfg(handle)
        self._grammar = nltk.PCFG.fromstring("\n".join(productions))
        self._config = config
        self._theorem_config = theory_generator.TheoremProverConfig(
            self._grammar,
            **config["theory"]["theorem_prover"],
        )

    def generate(self, index: int) -> dict[str, Any]:
        import numpy as np

        config = self._config
        for attempt in range(128):
            seed = _seed("ruletaker", index, attempt)
            random.seed(seed)
            np.random.seed(seed)
            example = self._module.generate_random_example(
                index,
                config.get("example_id_prefix", ""),
                self._grammar,
                self._theorem_config,
                config["theory"]["statement_types_per_example"],
                config["assertion"]["start_symbol"],
                "problog",
            )
            if example is None:
                continue
            payload = example.to_json()
            instance = payload["theory_assertion_instance"]
            english = payload["english"]
            return {
                "answer": bool(instance["label"]),
                "metadata": {
                    "attempt": attempt,
                    "minimum_proof_depth": instance["min_proof_depth"],
                    "proof_sha256": hashlib.sha256(
                        str(instance["proof"]).encode()
                    ).hexdigest(),
                    "seed": seed,
                    "theorem_prover": "problog",
                },
                "question": json.dumps(
                    {
                        "assertion": english["assertion_statement"],
                        "theory": english["theory_statements"],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
        raise RuntimeError(f"RuleTaker record {index} had no proved sample")


class _ProntoQA:
    distributions = ("numpy", "scipy")

    def __init__(self, root: Path):
        sys.path.insert(0, str(root))
        import run_experiment

        self._module = run_experiment

    def generate(self, index: int) -> dict[str, Any]:
        import numpy as np

        for attempt in range(128):
            seed = _seed("prontoqa", index, attempt)
            random.seed(seed)
            np.random.seed(seed)
            # ``generate_question`` mutates the module-global OntologyConfig.
            # Recreate the upstream default for every attempt so an ordinal is
            # independent of records generated earlier in the worker and can
            # be resumed directly.
            self._module.config = self._module.OntologyConfig(
                max_child_count=1,
                generate_negation=True,
                generate_properties=True,
                require_properties=False,
                stop_probability=0.3,
            )
            # Native generation is a rejection sampler whose valid-sample
            # probability drops sharply at larger depths.  Rotate the frozen
            # depth schedule across attempts so one unlucky deep ordinal does
            # not make the finite seed domain unusable.
            steps = 2 + ((index + attempt) % 5)
            output = self._module.generate_question(
                steps,
                None,
                formula_ordering="postorder",
                ontology="fictional",
                distractors="relevant",
                deduction_rule="ModusPonens",
                proofs_only=False,
            )
            question, query, _formulas, trace, answer, _proof = output
            if question is None:
                continue
            proof_payload = list(trace)
            return {
                "answer": str(answer),
                "metadata": {
                    "attempt": attempt,
                    "deduction_steps": steps,
                    # Native FOL nodes use the default object representation,
                    # which embeds process-specific memory addresses.  The
                    # native renderer's proof trace is the stable, exact proof
                    # artifact included in the emitted question.
                    "native_proof_trace_sha256": hashlib.sha256(
                        _canonical_bytes(proof_payload)
                    ).hexdigest(),
                    "seed": seed,
                },
                "question": json.dumps(
                    {
                        "premises": question,
                        "proof": list(trace),
                        "query": query,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
        raise RuntimeError(f"ProntoQA record {index} had no formal proof")


class _ReasoningGym:
    distributions = ("numpy",)
    _DATASETS = (
        "basic_arithmetic",
        "coin_flip",
        "knights_knaves",
        "propositional_logic",
        "shortest_path",
    )

    def __init__(self, root: Path):
        sys.path.insert(0, str(root))
        # The upstream package initializer eagerly imports every generator,
        # including unrelated optional solvers.  Register only the five frozen
        # native datasets used by this provider.  Namespace modules avoid
        # executing category initializers while preserving the exact staged
        # implementations in each imported leaf module.
        package = types.ModuleType("reasoning_gym")
        package.__path__ = [str(root / "reasoning_gym")]
        package.__package__ = "reasoning_gym"
        sys.modules["reasoning_gym"] = package
        # ``factory`` imports a coaching leaf, which would normally execute the
        # coaching package initializer first.  That initializer imports
        # ``experiment``, which imports the still-initializing factory and forms
        # a circular dependency.  Expose only the coaching primitives required
        # by the selected native datasets.
        coaching = types.ModuleType("reasoning_gym.coaching")
        coaching.__path__ = [str(root / "reasoning_gym" / "coaching")]
        coaching.__package__ = "reasoning_gym.coaching"
        sys.modules[coaching.__package__] = coaching
        attributes = importlib.import_module("reasoning_gym.coaching.attributes")
        base_curriculum = importlib.import_module(
            "reasoning_gym.coaching.base_curriculum"
        )
        for name in (
            "AttributeDefinition",
            "RangeAttributeDefinition",
            "ScalarAttributeDefinition",
        ):
            setattr(coaching, name, getattr(attributes, name))
        coaching.BaseCurriculum = base_curriculum.BaseCurriculum
        category_modules = {
            "arithmetic": "basic_arithmetic",
            "graphs": "shortest_path",
            "logic": ("knights_knaves", "propositional_logic"),
            "probability": "coin_flip",
        }
        for category in category_modules:
            module = types.ModuleType(f"reasoning_gym.{category}")
            module.__path__ = [str(root / "reasoning_gym" / category)]
            module.__package__ = f"reasoning_gym.{category}"
            sys.modules[module.__package__] = module
        factory = importlib.import_module("reasoning_gym.factory")
        self._dataset_modules = {}
        for category, names in category_modules.items():
            if isinstance(names, str):
                names = (names,)
            for name in names:
                self._dataset_modules[name] = importlib.import_module(
                    f"reasoning_gym.{category}.{name}"
                )
        self._factory = factory
        for name in self._DATASETS:
            if name not in factory.DATASETS:
                raise RuntimeError(f"Reasoning Gym dataset is absent: {name}")

    def generate(self, index: int) -> dict[str, Any]:
        dataset_name = self._DATASETS[index % len(self._DATASETS)]
        seed = _seed("reasoning_gym_exact_answer", index)
        dataset = self._factory.create_dataset(dataset_name, seed=seed, size=1)
        row = dataset[0]
        question = row.get("question")
        answer = row.get("answer")
        oracle_field = "answer"
        if answer is None and dataset_name == "propositional_logic":
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                raise RuntimeError(
                    "Reasoning Gym propositional_logic returned no metadata oracle"
                )
            answer = metadata.get("example_answer")
            premises = metadata.get("premises")
            module = self._dataset_modules[dataset_name]
            if (
                not isinstance(answer, str)
                or not answer
                or not isinstance(premises, list)
                or any(not isinstance(item, str) for item in premises)
            ):
                raise RuntimeError(
                    "Reasoning Gym propositional_logic metadata oracle is invalid"
                )
            parsed_answer = module.Expression.from_string(answer)
            parsed_premises = [module.Expression.from_string(item) for item in premises]
            if not dataset._is_valid_conclusion(
                parsed_premises, parsed_answer
            ) or dataset._is_trivial(parsed_answer):
                raise RuntimeError(
                    "Reasoning Gym propositional_logic metadata oracle "
                    "failed native truth-table validation"
                )
            oracle_field = "metadata.example_answer"
        if not isinstance(question, str) or not question or answer is None:
            raise RuntimeError(
                f"Reasoning Gym {dataset_name} returned no exact question/answer"
            )
        # The native scorer must accept its own oracle answer exactly.
        if oracle_field == "answer" and dataset.score_answer(str(answer), row) != 1.0:
            raise RuntimeError(
                f"Reasoning Gym {dataset_name} rejected its oracle answer"
            )
        return {
            "answer": str(answer),
            "metadata": {
                "dataset": dataset_name,
                "native_metadata": row.get("metadata", {}),
                "oracle_field": oracle_field,
                "seed": seed,
            },
            "question": question,
        }


_PROVIDER_TYPES = {
    "deepmind_mathematics_generator": _DeepMindMathematics,
    "clrs_text": _ClrsText,
    "ruletaker": _RuleTaker,
    "prontoqa": _ProntoQA,
    "reasoning_gym_exact_answer": _ReasoningGym,
}


def _worker(provider_name: str, source_dir: Path) -> int:
    try:
        provider_type = _PROVIDER_TYPES[provider_name]
        # Stdout is the JSONL protocol.  Native generators that print progress
        # or warnings are redirected to the worker log so they cannot corrupt
        # a handshake or record response.
        with contextlib.redirect_stdout(sys.stderr):
            provider = provider_type(source_dir)
        handshake = {
            "format": WORKER_FORMAT,
            "provider": provider_name,
            "ready": True,
            "runtime": _runtime_versions(provider.distributions),
            "source_dir": str(source_dir.resolve()),
        }
    # The worker boundary must serialize arbitrary third-party startup errors.
    except Exception as error:  # noqa: BLE001
        handshake = {
            "error": {
                "message": str(error),
                "type": type(error).__name__,
            },
            "format": WORKER_FORMAT,
            "provider": provider_name,
            "ready": False,
            "source_dir": str(source_dir.resolve()),
        }
        sys.stdout.buffer.write(_canonical_bytes(handshake))
        sys.stdout.buffer.flush()
        return 2

    sys.stdout.buffer.write(_canonical_bytes(handshake))
    sys.stdout.buffer.flush()
    for line_number, line in enumerate(sys.stdin.buffer, 1):
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or set(request) != {"index"}:
                raise ValueError("worker request fields must be exactly ['index']")
            index = request["index"]
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("worker index must be a non-negative integer")
            with contextlib.redirect_stdout(sys.stderr):
                record = provider.generate(index)
            if (
                not isinstance(record, dict)
                or set(record) != {"answer", "metadata", "question"}
                or not isinstance(record["question"], str)
                or not record["question"]
                or not isinstance(record["metadata"], dict)
            ):
                raise ValueError("native provider returned an invalid record")
            response = {"index": index, "record": record}
        # Keep one native provider failure inside the JSONL worker protocol.
        except Exception as error:  # noqa: BLE001
            response = {
                "error": {
                    "line": line_number,
                    "message": str(error),
                    "type": type(error).__name__,
                }
            }
        sys.stdout.buffer.write(_canonical_bytes(response))
        sys.stdout.buffer.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=PROCEDURAL_PROVIDERS, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.source_dir.is_dir() or args.source_dir.is_symlink():
        parser.error("--source-dir must be a regular directory")
    if os.environ.get("PYTHONHASHSEED") != "0":
        parser.error("objective workers require PYTHONHASHSEED=0")
    return _worker(args.provider, args.source_dir)


if __name__ == "__main__":
    raise SystemExit(main())
