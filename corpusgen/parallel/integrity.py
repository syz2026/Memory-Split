"""Ordered-stream and Merkle commitments for metadata-first schedules."""

from __future__ import annotations

import hashlib
import hmac

from .canonical import canonical_json_bytes
from .metadata import MetadataRecord
from .schedule import ScheduleRecord


def ordered_stream_commitments(
    schedule: tuple[ScheduleRecord, ...],
    metadata: tuple[MetadataRecord, ...],
) -> tuple[str, str]:
    """Return an order-sensitive stream hash and binary Merkle root."""

    by_id = {record.record_id: record for record in metadata}
    if len(by_id) != len(metadata):
        raise ValueError("metadata contains duplicate record ids")
    if len(schedule) != len(metadata):
        raise ValueError("schedule and metadata record counts differ")
    ordered_hasher = hashlib.sha256()
    leaves = []
    seen = set()
    for sequence, entry in enumerate(schedule):
        if entry.sequence != sequence:
            raise ValueError("schedule sequence is not contiguous")
        if entry.record_id in seen:
            raise ValueError(f"schedule contains duplicate record: {entry.record_id}")
        seen.add(entry.record_id)
        try:
            record = by_id[entry.record_id]
        except KeyError as error:
            raise ValueError(f"schedule record is missing metadata: {entry.record_id}") from error
        if (
            entry.lane != record.lane
            or entry.metadata_sha256 != record.metadata_sha256
            or entry.token_length != record.token_length
        ):
            raise ValueError(f"schedule metadata link drift: {entry.record_id}")
        leaf_payload = canonical_json_bytes(
            {
                "metadata_sha256": record.metadata_sha256,
                "record_id": record.record_id,
                "render_sha256": record.render_sha256,
                "sequence": sequence,
                "token_length": record.token_length,
            }
        )
        ordered_hasher.update(b"\x00")
        ordered_hasher.update(leaf_payload)
        leaves.append(hashlib.sha256(b"\x00" + leaf_payload).digest())
    if set(by_id) != seen:
        raise ValueError("schedule is missing metadata records")
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [
            hashlib.sha256(b"\x01" + leaves[index] + leaves[index + 1]).digest()
            for index in range(0, len(leaves), 2)
        ]
    return ordered_hasher.hexdigest(), leaves[0].hex()


def verify_stream_commitments(
    schedule: tuple[ScheduleRecord, ...],
    metadata: tuple[MetadataRecord, ...],
    *,
    ordered_stream_sha256: str,
    merkle_root_sha256: str,
) -> bool:
    """Verify both commitments, failing closed on either mismatch."""

    ordered, merkle = ordered_stream_commitments(schedule, metadata)
    if not hmac.compare_digest(ordered, ordered_stream_sha256):
        raise ValueError("ordered stream SHA-256 mismatch")
    if not hmac.compare_digest(merkle, merkle_root_sha256):
        raise ValueError("Merkle root SHA-256 mismatch")
    return True
