"""Metrics for the frozen, non-confirmatory Wikidata5M robustness study."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from corpusgen.wikidata_paths import (
    ADDRESS_EXCLUSION_REASONS,
    PATH_EXCLUSION_REASONS,
)


def _field(row: Mapping[str, Any] | object, name: str) -> Any:
    return row[name] if isinstance(row, Mapping) else getattr(row, name)


def _meta(row: Mapping[str, Any] | object) -> Mapping[str, Any]:
    value = _field(row, "meta")
    if not isinstance(value, Mapping):
        raise ValueError("Wikidata result metadata must be an object")
    return value


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or numerator < 0
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < 0
        or numerator > denominator
    ):
        raise ValueError("rate counts must satisfy 0 <= numerator <= denominator")
    return {
        "value": None if denominator == 0 else numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def selective_accuracy(rows) -> dict[str, int | float | None]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("selective accuracy requires a population")
    answered = [
        row for row in materialized if not bool(_field(row, "abstained"))
    ]
    correct = sum(bool(_field(row, "correct")) for row in answered)
    return {
        **_rate(correct, len(answered)),
        "population_denominator": len(materialized),
    }


def _pair_accuracy(rows: Sequence) -> dict[str, int | float | None]:
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        pair_id = str(_field(row, "pair_id"))
        variant = str(_field(row, "variant"))
        if variant not in {"original", "counterfactual"}:
            raise ValueError("pair variants must be original or counterfactual")
        if variant in grouped[pair_id]:
            raise ValueError("duplicate variant in Wikidata result pair")
        grouped[pair_id][variant] = row
    if any(set(pair) != {"original", "counterfactual"} for pair in grouped.values()):
        raise ValueError("every Wikidata pair requires both variants")
    correct = sum(
        bool(_field(pair["original"], "correct"))
        and bool(_field(pair["counterfactual"], "correct"))
        for pair in grouped.values()
    )
    return _rate(correct, len(grouped))


def _action_tuple(action: Any) -> tuple[Any, Any, Any, bool, bool]:
    if isinstance(action, Mapping):
        required = {
            "source_slot",
            "relation_id",
            "direction",
            "read",
            "halt",
        }
        if set(action) != required:
            raise ValueError("action fields do not match the frozen grammar")
        values = (
            action["source_slot"],
            action["relation_id"],
            action["direction"],
            action["read"],
            action["halt"],
        )
    else:
        values = tuple(action)
        if len(values) != 5:
            raise ValueError("serialized actions require five fields")
    return (
        values[0],
        values[1],
        values[2],
        bool(values[3]),
        bool(values[4]),
    )


def _transfer(rows: Sequence) -> dict[str, Any]:
    exact_paths = 0
    path_total = len(rows)
    all_slot_correct = 0
    all_slot_total = 0
    hop_counts: dict[int, CounterLike] = {}
    misses = 0
    gold_reads = 0
    read_attempts = 0

    for row in rows:
        predicted = [
            _action_tuple(value) for value in _field(row, "actions")
        ]
        gold = [
            _action_tuple(value) for value in _field(row, "gold_actions")
        ]
        predicted_all = [
            _action_tuple(value) for value in _field(row, "all_actions")
        ]
        gold_all = [
            _action_tuple(value) for value in _field(row, "gold_all_actions")
        ]
        referents = list(_field(row, "correct_referents"))
        if len(referents) != len(gold):
            raise ValueError("referent flags must match the gold read count")
        hop_count = _meta(row)["hop_count"]
        if (
            isinstance(hop_count, bool)
            or not isinstance(hop_count, int)
            or not 1 <= hop_count <= 6
            or len(gold) != hop_count
        ):
            raise ValueError("hop_count must match one through six gold reads")

        exact_paths += predicted == gold
        all_slot_total += len(gold_all)
        all_slot_correct += sum(
            index < len(predicted_all) and predicted_all[index] == expected
            for index, expected in enumerate(gold_all)
        )
        gold_reads += len(gold)
        miss_count = _field(row, "misses")
        if (
            isinstance(miss_count, bool)
            or not isinstance(miss_count, int)
            or miss_count < 0
        ):
            raise ValueError("misses must be a nonnegative integer")
        row_read_attempts = sum(action[3] for action in predicted)
        if miss_count > row_read_attempts:
            raise ValueError("misses cannot exceed actual read attempts")
        misses += miss_count
        read_attempts += row_read_attempts
        for index, expected in enumerate(gold):
            counts = hop_counts.setdefault(index + 1, CounterLike())
            actual = predicted[index] if index < len(predicted) else None
            counts.total += 1
            counts.action += actual == expected
            counts.relation += (
                actual is not None and actual[1] == expected[1]
            )
            counts.direction += (
                actual is not None and actual[2] == expected[2]
            )
            counts.referent += bool(referents[index])

    per_hop = {
        str(index): {
            "action": _rate(counts.action, counts.total),
            "relation": _rate(counts.relation, counts.total),
            "direction": _rate(counts.direction, counts.total),
            "referent": _rate(counts.referent, counts.total),
        }
        for index, counts in sorted(hop_counts.items())
    }
    return {
        "exact_action_path": _rate(exact_paths, path_total),
        "all_action_slots": _rate(all_slot_correct, all_slot_total),
        "per_hop": per_hop,
        "miss_rate": _rate(misses, read_attempts),
        "misses": misses,
        "read_attempts": read_attempts,
        "gold_reads": gold_reads,
    }


class CounterLike:
    __slots__ = ("action", "direction", "referent", "relation", "total")

    def __init__(self) -> None:
        self.action = 0
        self.direction = 0
        self.referent = 0
        self.relation = 0
        self.total = 0


def _mode_metrics(rows: Sequence) -> dict[str, Any]:
    if not rows:
        raise ValueError("each store mode requires result rows")
    total = len(rows)
    correct = sum(bool(_field(row, "correct")) for row in rows)
    abstained = sum(bool(_field(row, "abstained")) for row in rows)
    return {
        "accuracy": _rate(correct, total),
        "coverage": _rate(total - abstained, total),
        "abstention": _rate(abstained, total),
        "selective_accuracy": selective_accuracy(rows),
        "pair_accuracy": _pair_accuracy(rows),
        "transfer": _transfer(rows),
        "n_rows": total,
    }


def _slice_value(row, name: str) -> str:
    meta = _meta(row)
    if name == "hop":
        return str(meta["hop_count"])
    if name == "task":
        return str(_field(row, "task"))
    if name == "composition_path":
        value = meta.get("relation_path_hash")
        if not isinstance(value, str) or not value:
            raise ValueError("missing relation_path_hash metadata")
        return value
    key = f"{name}_slice"
    value = meta.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing {key} metadata")
    return value


def _slice_metrics(rows: Sequence, name: str) -> dict[str, Any]:
    grouped: dict[str, list] = defaultdict(list)
    for row in rows:
        grouped[_slice_value(row, name)].append(row)
    return {
        key: _mode_metrics(group)
        for key, group in sorted(grouped.items())
    }


def _survival_measurement(
    coverage: Mapping[str, Any],
    name: str,
) -> dict[str, int | float | None]:
    value = coverage[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} survival must be an object")
    candidates = value["candidates"]
    surviving = value["surviving"]
    measurement = _rate(surviving, candidates)
    if "exclusions" in value:
        exclusions = value["exclusions"]
        if not isinstance(exclusions, Mapping):
            raise ValueError(f"{name} exclusions must be an object")
        if sum(exclusions.values()) + surviving != candidates:
            raise ValueError(f"{name} candidate accounting is not exhaustive")
        measurement["exclusions"] = dict(sorted(exclusions.items()))
    return measurement


def _exclusion_accounting(
    coverage: Mapping[str, Any],
    name: str,
    required_reasons: Sequence[str],
) -> dict[str, Any]:
    value = coverage.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} accounting must be an object")
    candidates = value.get("candidates")
    surviving = value.get("surviving")
    exclusions = value.get("exclusions")
    if not isinstance(exclusions, Mapping):
        raise ValueError(f"{name} exclusions must be an object")
    if set(exclusions) != set(required_reasons):
        raise ValueError(
            f"{name} exclusions do not contain every frozen reason"
        )
    if (
        isinstance(candidates, bool)
        or not isinstance(candidates, int)
        or candidates < 0
        or isinstance(surviving, bool)
        or not isinstance(surviving, int)
        or surviving < 0
    ):
        raise ValueError(f"{name} accounting counts must be nonnegative")
    counts = {}
    for reason in required_reasons:
        count = exclusions[reason]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{name} exclusion counts must be nonnegative")
        counts[reason] = count
    excluded = sum(counts.values())
    if candidates != surviving + excluded:
        raise ValueError(
            f"{name} candidate accounting is not exhaustive"
        )
    output = {
        "survival": _rate(surviving, candidates),
        "excluded": _rate(excluded, candidates),
        "exclusions": {
            reason: _rate(counts[reason], candidates)
            for reason in required_reasons
        },
        "invariant": {
            "candidates": candidates,
            "surviving": surviving,
            "excluded": excluded,
            "candidate_equals_survivor_plus_exclusions": True,
        },
    }
    per_hop = value.get("per_hop")
    if per_hop is not None:
        if not isinstance(per_hop, Mapping):
            raise ValueError(f"{name} per_hop accounting must be an object")
        output["per_hop"] = {
            str(hop): _exclusion_accounting(
                {"hop": accounting},
                "hop",
                required_reasons,
            )
            for hop, accounting in sorted(
                per_hop.items(),
                key=lambda item: int(item[0]),
            )
        }
        if sum(
            item["invariant"]["candidates"]
            for item in output["per_hop"].values()
        ) != candidates:
            raise ValueError("per-hop path candidates do not match total")
        if sum(
            item["invariant"]["surviving"]
            for item in output["per_hop"].values()
        ) != surviving:
            raise ValueError("per-hop path survivors do not match total")
        for reason in required_reasons:
            if sum(
                item["exclusions"][reason]["numerator"]
                for item in output["per_hop"].values()
            ) != counts[reason]:
                raise ValueError(
                    f"per-hop {reason} exclusions do not match total"
                )
    return output


def compute_wikidata_metrics(rows, coverage) -> dict[str, Any]:
    """Compute transfer metrics without producing a confirmatory decision."""

    materialized = list(rows)
    if not materialized:
        raise ValueError("Wikidata metrics require result rows")
    by_mode: dict[str, list] = defaultdict(list)
    seen = set()
    for row in materialized:
        memory = str(_field(row, "memory"))
        if memory not in {"on", "off"}:
            raise ValueError("memory mode must be on or off")
        qid = str(_field(row, "qid"))
        identity = (memory, qid)
        if identity in seen:
            raise ValueError("duplicate Wikidata result row")
        seen.add(identity)
        by_mode[memory].append(row)
    if set(by_mode) != {"on", "off"}:
        raise ValueError("Wikidata results require store ON and OFF rows")
    on_ids = {str(_field(row, "qid")) for row in by_mode["on"]}
    off_ids = {str(_field(row, "qid")) for row in by_mode["off"]}
    if on_ids != off_ids:
        raise ValueError("store ON and OFF must evaluate identical items")

    coverage_value = (
        coverage.to_dict() if hasattr(coverage, "to_dict") else coverage
    )
    if not isinstance(coverage_value, Mapping):
        raise TypeError("coverage must be a manifest or mapping")
    store = {
        mode: _mode_metrics(by_mode[mode])
        for mode in ("on", "off")
    }
    slices = {
        name: {
            mode: _slice_metrics(by_mode[mode], name)
            for mode in ("on", "off")
        }
        for name in (
            "alias",
            "composition",
            "composition_path",
            "hop",
            "task",
        )
    }
    return {
        "analysis_role": "robustness_only",
        "confirmatory_verdict_eligible": False,
        "artifact_survival": {
            name: _survival_measurement(coverage_value, name)
            for name in ("entities", "relations", "addresses", "paths")
        },
        "exclusion_accounting": {
            "address": _exclusion_accounting(
                coverage_value,
                "addresses",
                ADDRESS_EXCLUSION_REASONS,
            ),
            "path": _exclusion_accounting(
                coverage_value,
                "paths",
                PATH_EXCLUSION_REASONS,
            ),
        },
        "store": store,
        "store_on_minus_off_accuracy": (
            store["on"]["accuracy"]["value"]
            - store["off"]["accuracy"]["value"]
        ),
        "slices": slices,
        "n_rows": len(materialized),
    }
