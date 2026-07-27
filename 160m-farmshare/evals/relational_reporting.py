"""Deterministic publication for the protected relational analysis.

The confirmatory decision document is built only from typed verdict inputs.
Supporting, exploratory, and robustness data are serialized separately and
cannot enter the decision hash.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from evals.figures import render_plot_svg
from evals.relational_contracts import (
    _rename_directory_noreplace_between,
    canonical_json_bytes,
)
from evals.relational_metrics import EXPECTED_TASKS
from evals.relational_stats import (
    BOOTSTRAP_VERSION,
    CONFIRMATORY_SEEDS,
    FROZEN_N_BOOT,
    FROZEN_PERCENTILE_INDICES,
    MAX_BOOTSTRAP_CHUNK,
    VerdictInputs,
    decide_verdict,
    verdict_inputs_to_dict,
)


REPORT_SECTIONS = (
    "paired_deltas",
    "dose_interaction",
    "memory_factorial",
    "controls_by_hop_composition",
    "guardrails",
    "wikidata_robustness",
)
_SECTION_ROLES = MappingProxyType(
    {
        "paired_deltas": "confirmatory",
        "dose_interaction": "confirmatory",
        "memory_factorial": "supporting_only",
        "controls_by_hop_composition": "supporting_only",
        "guardrails": "instrument_only",
        "wikidata_robustness": "robustness_only",
    }
)
_INPUT_BINDING_FIELDS = {
    "runs_root_sha256",
    "preregistration_sha256",
    "analysis_code_sha256",
    "guardrail_receipt_sha256",
}
_BOOTSTRAP_FIELDS = {
    "version",
    "n_boot",
    "rng_seed",
    "chunk_size",
    "percentile_indices",
}
_SHA256_LENGTH = 64


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _hash_record(value: Any) -> str:
    return _sha256(canonical_json_bytes(value))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return str(value)


def _canonical_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def _contains_forbidden_confirmatory_value(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.casefold()
        return (
            "selective" in normalized
            or normalized in {"robustness_only", "exploratory_only"}
        )
    if isinstance(value, Mapping):
        return any(
            _contains_forbidden_confirmatory_value(key)
            or _contains_forbidden_confirmatory_value(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_confirmatory_value(item) for item in value)
    return False


@dataclass(frozen=True)
class AnalysisSection:
    analysis_role: str
    rows: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.analysis_role, str)
            or not self.analysis_role
        ):
            raise ValueError("analysis role must be a non-empty string")
        if isinstance(self.rows, (str, bytes)) or not isinstance(
            self.rows, Sequence
        ):
            raise TypeError("analysis section rows must be a sequence")
        materialized = []
        for row in self.rows:
            if not isinstance(row, Mapping):
                raise TypeError("analysis section rows must contain mappings")
            copied = _canonical_copy(row)
            if not isinstance(copied, dict):
                raise ValueError("analysis section row must be an object")
            materialized.append(MappingProxyType(copied))
        object.__setattr__(self, "rows", tuple(materialized))


def _validate_input_bindings(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _INPUT_BINDING_FIELDS:
        raise ValueError("input binding fields are not exact")
    receipts = raw["guardrail_receipt_sha256"]
    if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
        raise ValueError("guardrail receipt input binding must be a sequence")
    receipt_hashes = [
        _require_sha256(value, "guardrail receipt input binding")
        for value in receipts
    ]
    if not receipt_hashes or len(receipt_hashes) != len(set(receipt_hashes)):
        raise ValueError(
            "guardrail receipt input bindings must be non-empty and unique"
        )
    return {
        "runs_root_sha256": _require_sha256(
            raw["runs_root_sha256"], "runs root input binding"
        ),
        "preregistration_sha256": _require_sha256(
            raw["preregistration_sha256"], "preregistration input binding"
        ),
        "analysis_code_sha256": _require_sha256(
            raw["analysis_code_sha256"], "analysis code input binding"
        ),
        "guardrail_receipt_sha256": sorted(receipt_hashes),
    }


def _validate_bootstrap_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _BOOTSTRAP_FIELDS:
        raise ValueError("bootstrap configuration fields are not exact")
    rng_seed = raw["rng_seed"]
    if (
        isinstance(rng_seed, bool)
        or not isinstance(rng_seed, int)
        or rng_seed < 0
    ):
        raise ValueError("bootstrap rng_seed must be a non-negative integer")
    if (
        raw["version"] != BOOTSTRAP_VERSION
        or raw["n_boot"] != FROZEN_N_BOOT
        or raw["chunk_size"] != MAX_BOOTSTRAP_CHUNK
        or raw["percentile_indices"] != list(FROZEN_PERCENTILE_INDICES)
    ):
        raise ValueError(
            "bootstrap config must use 10,000 frozen hierarchical replicates"
        )
    return {
        "version": BOOTSTRAP_VERSION,
        "n_boot": FROZEN_N_BOOT,
        "rng_seed": rng_seed,
        "chunk_size": MAX_BOOTSTRAP_CHUNK,
        "percentile_indices": list(FROZEN_PERCENTILE_INDICES),
    }


def _validate_run_matrix(
    raw: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("run matrix must be a sequence")
    rows = []
    identities = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("run matrix entries must be objects")
        copied = _canonical_copy(item)
        required = {"model", "arm", "load", "seed", "checkpoint_sha256"}
        if not required <= set(copied):
            raise ValueError("run matrix entry is missing an identity field")
        if (
            not all(
                isinstance(copied[name], str) and copied[name]
                for name in ("model", "arm", "load")
            )
            or isinstance(copied["seed"], bool)
            or not isinstance(copied["seed"], int)
        ):
            raise ValueError("run matrix identity is invalid")
        _require_sha256(
            copied["checkpoint_sha256"], "run matrix checkpoint SHA-256"
        )
        identity = (
            copied["model"],
            copied["arm"],
            copied["load"],
            copied["seed"],
        )
        if identity in identities:
            raise ValueError("run matrix contains a duplicate identity")
        identities.add(identity)
        rows.append(copied)
    return sorted(
        rows,
        key=lambda item: (
            item["model"],
            item["arm"],
            item["load"],
            item["seed"],
        ),
    )


def _validate_secondary_analyses(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("secondary analyses must be an object")
    copied = _canonical_copy(raw)
    for name, value in copied.items():
        if not isinstance(value, dict):
            raise ValueError(f"secondary analysis {name!r} must be an object")
        if value.get("analysis_role") not in {
            "supporting_only",
            "exploratory_only",
            "robustness_only",
        }:
            raise ValueError(
                f"secondary analysis {name!r} has an invalid analysis role"
            )
        if value.get("confirmatory_verdict_eligible") is not False:
            raise ValueError(
                f"secondary analysis {name!r} must be verdict-ineligible"
            )
    return copied


def build_analysis_document(
    *,
    verdict_inputs: VerdictInputs,
    input_bindings: Mapping[str, Any],
    run_matrix: Sequence[Mapping[str, Any]],
    bootstrap_config: Mapping[str, Any],
    secondary_analyses: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical decision document without secondary-data leakage."""

    if not isinstance(verdict_inputs, VerdictInputs):
        raise TypeError("analysis document requires typed VerdictInputs")
    inputs = verdict_inputs_to_dict(verdict_inputs)
    verdict = decide_verdict(verdict_inputs)
    bindings = _validate_input_bindings(input_bindings)
    matrix = _validate_run_matrix(run_matrix)
    bootstrap = _validate_bootstrap_config(bootstrap_config)
    secondary = _validate_secondary_analyses(secondary_analyses)
    decision_payload = {
        "record_type": "relational_confirmatory_decision",
        "schema_version": 1,
        "seeds": list(CONFIRMATORY_SEEDS),
        "verdict": verdict,
        "verdict_inputs": inputs,
        "input_bindings": bindings,
        "run_matrix": matrix,
        "bootstrap_config": bootstrap,
    }
    return {
        "record_type": "relational_analysis",
        "schema_version": 1,
        "analysis_role": "confirmatory",
        "seeds": list(CONFIRMATORY_SEEDS),
        "verdict": verdict,
        "verdict_inputs": inputs,
        "decision_sha256": _hash_record(decision_payload),
        "input_bindings": bindings,
        "run_matrix": matrix,
        "bootstrap_config": bootstrap,
        "secondary_analyses": secondary,
        "artifacts": {
            "plot_data": {},
            "figures": {},
            "tables": {},
        },
    }


def canonical_analysis_bytes(value: Mapping[str, Any]) -> bytes:
    if not isinstance(value, Mapping):
        raise TypeError("analysis document must be a mapping")
    return canonical_json_bytes(value)


def _finite_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _serialized_estimate(
    raw: object,
    name: str,
) -> dict[str, Any]:
    fields = {
        "mean",
        "ci_lo",
        "ci_hi",
        "seed_deltas",
        "cohen_dz",
        "effect_note",
    }
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError(f"{name} estimate fields are not exact")
    mean = _finite_number(raw["mean"], f"{name} mean")
    low = _finite_number(raw["ci_lo"], f"{name} lower bound")
    high = _finite_number(raw["ci_hi"], f"{name} upper bound")
    if not low <= mean <= high:
        raise ValueError(f"{name} interval does not contain its mean")
    seed_deltas_raw = raw["seed_deltas"]
    if (
        not isinstance(seed_deltas_raw, list)
        or len(seed_deltas_raw) != len(CONFIRMATORY_SEEDS)
    ):
        raise ValueError(f"{name} requires five seed deltas")
    seed_deltas = [
        _finite_number(value, f"{name} seed delta")
        for value in seed_deltas_raw
    ]
    if not math.isclose(
        sum(seed_deltas) / len(seed_deltas),
        mean,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"{name} mean disagrees with seed deltas")
    cohen_dz = raw["cohen_dz"]
    if cohen_dz is not None:
        _finite_number(cohen_dz, f"{name} Cohen dz")
    if not isinstance(raw["effect_note"], str) or not raw["effect_note"]:
        raise ValueError(f"{name} effect note is invalid")
    return raw


def _verdict_from_serialized_inputs(raw: object) -> str:
    fields = {
        "split_dense_360",
        "split_dense_160_high",
        "dose_interaction_160",
        "split_random_160_high",
        "task_means_360",
        "task_means_160_high",
        "guardrail_report_sha256",
        "all_guardrails_passed",
    }
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError("serialized verdict input fields are not exact")
    estimates = {
        name: _serialized_estimate(raw[name], name)
        for name in (
            "split_dense_360",
            "split_dense_160_high",
            "dose_interaction_160",
            "split_random_160_high",
        )
    }
    task_means = {}
    for name in ("task_means_360", "task_means_160_high"):
        values = raw[name]
        if not isinstance(values, dict) or set(values) != set(EXPECTED_TASKS):
            raise ValueError(f"{name} does not match the frozen tasks")
        task_means[name] = [
            _finite_number(values[task], f"{name}.{task}")
            for task in EXPECTED_TASKS
        ]
    report_hashes = raw["guardrail_report_sha256"]
    if (
        not isinstance(report_hashes, list)
        or not report_hashes
        or len(report_hashes) != len(set(report_hashes))
    ):
        raise ValueError("guardrail report hashes are invalid")
    for value in report_hashes:
        _require_sha256(value, "guardrail report SHA-256")
    all_guardrails = raw["all_guardrails_passed"]
    if not isinstance(all_guardrails, bool):
        raise ValueError("all_guardrails_passed must be Boolean")
    if not all_guardrails:
        return "invalid"

    validates = (
        estimates["split_dense_360"]["mean"] >= 0.02
        and estimates["split_dense_360"]["ci_lo"] > 0.0
        and estimates["split_dense_160_high"]["ci_lo"] > 0.0
        and estimates["dose_interaction_160"]["ci_lo"] > 0.0
        and estimates["split_random_160_high"]["ci_lo"] > 0.0
        and all(value > 0.0 for value in task_means["task_means_360"])
        and all(
            value > 0.0
            for value in task_means["task_means_160_high"]
        )
    )
    if validates:
        return "validated"
    practical_null = (
        estimates["split_dense_160_high"]["ci_hi"] < 0.02
        and estimates["split_dense_360"]["ci_hi"] < 0.02
        and estimates["dose_interaction_160"]["ci_hi"] <= 0.0
        and estimates["split_random_160_high"]["ci_hi"] <= 0.01
    )
    return "practical_null" if practical_null else "inconclusive"


def _validate_analysis_document(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("analysis document must be an object")
    expected = {
        "record_type",
        "schema_version",
        "analysis_role",
        "seeds",
        "verdict",
        "verdict_inputs",
        "decision_sha256",
        "input_bindings",
        "run_matrix",
        "bootstrap_config",
        "secondary_analyses",
        "artifacts",
    }
    if set(raw) != expected:
        raise ValueError("analysis document fields are not exact")
    copied = _canonical_copy(raw)
    if (
        copied["record_type"] != "relational_analysis"
        or copied["schema_version"] != 1
        or copied["analysis_role"] != "confirmatory"
        or copied["seeds"] != list(CONFIRMATORY_SEEDS)
        or copied["verdict"]
        not in {"validated", "practical_null", "inconclusive", "invalid"}
    ):
        raise ValueError("analysis document identity is invalid")
    _require_sha256(copied["decision_sha256"], "decision SHA-256")
    copied["input_bindings"] = _validate_input_bindings(
        copied["input_bindings"]
    )
    copied["run_matrix"] = _validate_run_matrix(copied["run_matrix"])
    copied["bootstrap_config"] = _validate_bootstrap_config(
        copied["bootstrap_config"]
    )
    copied["secondary_analyses"] = _validate_secondary_analyses(
        copied["secondary_analyses"]
    )
    artifacts = copied["artifacts"]
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != {"plot_data", "figures", "tables"}
        or any(not isinstance(values, dict) for values in artifacts.values())
    ):
        raise ValueError("analysis artifact manifest is invalid")
    decision_payload = {
        "record_type": "relational_confirmatory_decision",
        "schema_version": 1,
        "seeds": copied["seeds"],
        "verdict": copied["verdict"],
        "verdict_inputs": copied["verdict_inputs"],
        "input_bindings": copied["input_bindings"],
        "run_matrix": copied["run_matrix"],
        "bootstrap_config": copied["bootstrap_config"],
    }
    if copied["decision_sha256"] != _hash_record(decision_payload):
        raise ValueError("analysis decision hash mismatch")
    if copied["verdict"] != _verdict_from_serialized_inputs(
        copied["verdict_inputs"]
    ):
        raise ValueError("analysis verdict disagrees with frozen decision rules")
    return copied


def _validate_sections(
    sections: Mapping[str, AnalysisSection],
) -> dict[str, AnalysisSection]:
    if not isinstance(sections, Mapping) or set(sections) != set(
        REPORT_SECTIONS
    ):
        raise ValueError("report section set is not exact")
    validated = {}
    for name in REPORT_SECTIONS:
        section = sections[name]
        if not isinstance(section, AnalysisSection):
            raise TypeError("report sections must use AnalysisSection")
        if section.analysis_role != _SECTION_ROLES[name]:
            raise ValueError(
                f"{name} analysis role must be {_SECTION_ROLES[name]}"
            )
        if (
            section.analysis_role == "confirmatory"
            and _contains_forbidden_confirmatory_value(section.rows)
        ):
            raise ValueError(
                "Selective or robustness data cannot enter "
                "confirmatory report sections"
            )
        validated[name] = section
    return validated


def _scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("table values must be finite")
        return format(value, ".17g")
    if isinstance(value, (dict, list, tuple)):
        return canonical_json_bytes(value).decode().rstrip("\n")
    return str(value)


def _table_columns(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({str(key) for row in rows for key in row}))


def _render_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    columns = _table_columns(rows)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_scalar_text(row.get(column)) for column in columns])
    return stream.getvalue().encode()


def _markdown_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _render_markdown(
    name: str,
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    columns = _table_columns(rows)
    lines = [f"# {name.replace('_', ' ').title()}", ""]
    if not columns:
        lines.append("_No observations._")
    else:
        lines.extend(
            [
                "| " + " | ".join(columns) + " |",
                "| " + " | ".join("---" for _ in columns) + " |",
            ]
        )
        lines.extend(
            "| "
            + " | ".join(
                _markdown_escape(_scalar_text(row.get(column)))
                for column in columns
            )
            + " |"
            for row in rows
        )
    return ("\n".join(lines) + "\n").encode()


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _render_latex(
    name: str,
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    columns = _table_columns(rows)
    if not columns:
        return (
            "\\begin{table}\n"
            f"\\caption{{{_latex_escape(name.replace('_', ' ').title())}}}\n"
            "\\begin{tabular}{l}\nNo observations\\\\\n"
            "\\end{tabular}\n\\end{table}\n"
        ).encode()
    lines = [
        "\\begin{table}",
        f"\\caption{{{_latex_escape(name.replace('_', ' ').title())}}}",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        " & ".join(_latex_escape(column) for column in columns) + r"\\",
        r"\hline",
    ]
    lines.extend(
        " & ".join(
            _latex_escape(_scalar_text(row.get(column)))
            for column in columns
        )
        + r"\\"
        for row in rows
    )
    lines.extend(["\\end{tabular}", "\\end{table}"])
    return ("\n".join(lines) + "\n").encode()


def required_report_files() -> tuple[str, ...]:
    files = ["analysis.json"]
    for name in REPORT_SECTIONS:
        files.extend(
            (
                f"tables/{name}.csv",
                f"tables/{name}.md",
                f"tables/{name}.tex",
                f"figures/{name}.plot-data.json",
                f"figures/{name}.svg",
            )
        )
    return tuple(sorted(files))


def _contains_parent_reference(path: Path) -> bool:
    return any(part == ".." for part in path.parts)


def _regular_parent(destination: Path) -> Path:
    if _contains_parent_reference(destination):
        raise ValueError("analysis output path cannot contain traversal")
    parent = destination.parent
    if parent.is_symlink():
        raise ValueError("analysis output cannot use a symlink parent")
    if not parent.is_dir():
        raise ValueError("analysis output parent must be a regular directory")
    if parent.resolve(strict=True) != parent.absolute():
        raise ValueError("analysis output cannot traverse symlink components")
    return parent


def _write_synced(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def publish_analysis_bundle(
    destination: str | Path,
    *,
    analysis: Mapping[str, Any],
    sections: Mapping[str, AnalysisSection],
) -> Path:
    """Atomically publish one deterministic, fail-closed report bundle."""

    output = Path(destination)
    parent = _regular_parent(output)
    if os.path.lexists(output):
        raise FileExistsError(f"analysis output already exists: {output}")
    document = _validate_analysis_document(analysis)
    validated_sections = _validate_sections(sections)
    if any(document["artifacts"][kind] for kind in document["artifacts"]):
        raise ValueError("analysis artifact manifest must be empty before publish")

    stage = Path(
        tempfile.mkdtemp(dir=parent, prefix=f".{output.name}.stage-")
    )
    try:
        artifact_hashes: dict[str, dict[str, str]] = {
            "plot_data": {},
            "figures": {},
            "tables": {},
        }
        for name in REPORT_SECTIONS:
            section = validated_sections[name]
            rows = [dict(row) for row in section.rows]
            sidecar_relative = f"figures/{name}.plot-data.json"
            sidecar = canonical_json_bytes(
                {
                    "record_type": "relational_plot_data",
                    "schema_version": 1,
                    "section": name,
                    "analysis_role": section.analysis_role,
                    "rows": rows,
                }
            )
            _write_synced(stage / sidecar_relative, sidecar)
            artifact_hashes["plot_data"][sidecar_relative] = _sha256(sidecar)

            figure_relative = f"figures/{name}.svg"
            render_plot_svg(name, rows, stage / figure_relative)
            figure_content = (stage / figure_relative).read_bytes()
            artifact_hashes["figures"][figure_relative] = _sha256(
                figure_content
            )

            table_contents = {
                f"tables/{name}.csv": _render_csv(rows),
                f"tables/{name}.md": _render_markdown(name, rows),
                f"tables/{name}.tex": _render_latex(name, rows),
            }
            for relative, content in table_contents.items():
                _write_synced(stage / relative, content)
                artifact_hashes["tables"][relative] = _sha256(content)

        document["artifacts"] = artifact_hashes
        _write_synced(
            stage / "analysis.json",
            canonical_analysis_bytes(document),
        )
        if {
            path.relative_to(stage).as_posix()
            for path in stage.rglob("*")
            if path.is_file()
        } != set(required_report_files()):
            raise AssertionError("staged report file set is not exact")
        stage_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            _rename_directory_noreplace_between(
                parent_fd,
                stage.name,
                parent_fd,
                output.name,
            )
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return output


def validate_analysis_bundle(root: str | Path) -> dict[str, Any]:
    bundle = Path(root)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("analysis bundle must be a regular directory")
    if bundle.resolve(strict=True) != bundle.absolute():
        raise ValueError("analysis bundle cannot traverse symlinks")
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != set(required_report_files()):
        raise ValueError("analysis bundle file set is not exact")
    analysis_path = bundle / "analysis.json"
    content = analysis_path.read_bytes()
    try:
        parsed = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("analysis JSON is malformed") from exc
    if content != canonical_analysis_bytes(parsed):
        raise ValueError("analysis JSON is not canonical")
    document = _validate_analysis_document(parsed)
    for kind, expected_files in document["artifacts"].items():
        expected_kind_files = {
            relative
            for relative in required_report_files()
            if (
                (kind == "plot_data" and relative.endswith(".plot-data.json"))
                or (kind == "figures" and relative.endswith(".svg"))
                or (kind == "tables" and relative.startswith("tables/"))
            )
        }
        if set(expected_files) != expected_kind_files:
            raise ValueError(f"analysis {kind} artifact set is not exact")
        for relative, expected_hash in expected_files.items():
            _require_sha256(expected_hash, f"{kind} artifact hash")
            path = bundle / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"{kind} artifact is not a regular file")
            if _sha256(path.read_bytes()) != expected_hash:
                raise ValueError(f"{kind} artifact hash mismatch")

    for name in REPORT_SECTIONS:
        sidecar = bundle / f"figures/{name}.plot-data.json"
        sidecar_content = sidecar.read_bytes()
        try:
            value = json.loads(sidecar_content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("plot sidecar is malformed") from exc
        if sidecar_content != canonical_json_bytes(value):
            raise ValueError("plot sidecar is not canonical")
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "record_type",
                "schema_version",
                "section",
                "analysis_role",
                "rows",
            }
            or value["record_type"] != "relational_plot_data"
            or value["schema_version"] != 1
            or value["section"] != name
            or value["analysis_role"] != _SECTION_ROLES[name]
            or not isinstance(value["rows"], list)
        ):
            raise ValueError("plot sidecar contract is invalid")
        if (
            value["analysis_role"] == "confirmatory"
            and _contains_forbidden_confirmatory_value(value["rows"])
        ):
            raise ValueError("plot sidecar leaks selective secondary data")
    return document
