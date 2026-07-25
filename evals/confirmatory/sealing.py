"""Content addressing for the pre-launch sealed evaluator fixture."""

from __future__ import annotations

from collections.abc import Mapping
import re

from evals.confirmatory.contracts import canonical_sha256


SEALED_FIXTURE_SCHEMA = "memorysplit.confirmatory.sealed-fixture.v3"
SEALED_FIXTURE_MEMBERS = (
    "items.jsonl",
    "sealed-gold.jsonl",
    "stores.jsonl",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sealed_fixture_sha256(members: Mapping[str, str]) -> str:
    """Hash one exact, ordered three-member fixture inventory."""

    if not isinstance(members, Mapping) or set(members) != set(
        SEALED_FIXTURE_MEMBERS
    ):
        raise ValueError("sealed fixture member inventory is not exact")
    inventory: dict[str, str] = {}
    for name in SEALED_FIXTURE_MEMBERS:
        digest = members[name]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"sealed fixture member hash is invalid: {name}")
        inventory[name] = digest
    return canonical_sha256(
        {
            "record_type": SEALED_FIXTURE_SCHEMA,
            "schema_version": 3,
            "members": inventory,
        }
    )
