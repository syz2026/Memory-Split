from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import shutil
import struct
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from corpusgen.graph_records import GraphAddress, GraphRow
from corpusgen.relation_codec import RelationCodec
from organizer.graph_store import StoreStats

_SCHEMA_VERSION = 1
_HASH_ALGORITHM = "blake2b-64-memsplit-v1"
_UINT64_MAX = (1 << 64) - 1
_INDEX_STRUCT = struct.Struct("<QQ")
_KEY_STRUCT = struct.Struct("<QHB")
_ROW_STRUCT = struct.Struct("<QHBBQQQQQQ")
_FILE_NAMES = ("index.bin", "rows.bin", "blobs.bin")
_MANIFEST_FIELDS = {
    "schema_version",
    "hash_algorithm",
    "codec_sha256",
    "row_count",
    "capacity",
    "snapshot_sha256",
    "file_sha256",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_DIRECTION_TO_CODE = {"out": 0, "in": 1}
_CODE_TO_DIRECTION = {value: key for key, value in _DIRECTION_TO_CODE.items()}
_TARGET_KIND_TO_CODE = {"entity": 0, "literal": 1}
_CODE_TO_TARGET_KIND = {
    value: key for key, value in _TARGET_KIND_TO_CODE.items()
}

def _stable_hash(key: bytes) -> int:
    digest = hashlib.blake2b(
        key,
        digest_size=8,
        person=b"memsplit",
    ).digest()
    return int.from_bytes(digest, "little")


def _checked_hash(key: bytes) -> int:
    value = _stable_hash(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("graph key hash must be an integer")
    if not 0 <= value <= _UINT64_MAX:
        raise ValueError("graph key hash must fit uint64")
    return value


def _key_bytes(source_id: int, relation_index: int, direction_code: int) -> bytes:
    return _KEY_STRUCT.pack(source_id, relation_index, direction_code)


def _capacity_for_rows(row_count: int) -> int:
    capacity = 1
    while row_count * 10 > capacity * 7:
        capacity *= 2
    return capacity


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )


def _canonical_row_bytes(row: GraphRow) -> bytes:
    return json.dumps(
        row.as_json(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _qualifier_bytes(qualifiers: tuple[tuple[str, str], ...]) -> bytes:
    return json.dumps(
        [list(qualifier) for qualifier in qualifiers],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot read packed graph file: {path.name}") from exc
    return digest.hexdigest()


def _write_file(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _require_uint64(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _UINT64_MAX
    ):
        raise ValueError(f"{name} must be a uint64")
    return value


def _entity_target(target: str) -> int:
    if not target.isascii() or not target.isdecimal():
        raise ValueError("entity graph targets must be canonical uint64 strings")
    value = int(target)
    if value > _UINT64_MAX or str(value) != target:
        raise ValueError("entity graph targets must be canonical uint64 strings")
    return value


def _validated_qualifiers(row: GraphRow) -> tuple[tuple[str, str], ...]:
    qualifiers = row.qualifiers
    if not isinstance(qualifiers, tuple) or any(
        not isinstance(qualifier, tuple)
        or len(qualifier) != 2
        or not all(isinstance(value, str) for value in qualifier)
        for qualifier in qualifiers
    ):
        raise ValueError("graph row qualifiers must be string pairs")
    return qualifiers


def _materialize_rows(
    rows: Iterable[GraphRow],
    relation_indices: Mapping[str, int],
) -> tuple[GraphRow, ...]:
    materialized = []
    for row in rows:
        if not isinstance(row, GraphRow):
            raise TypeError("packed graph rows must be GraphRow values")
        _require_uint64(row.source_id, "graph source_id")
        if row.relation_id not in relation_indices:
            raise ValueError(
                f"graph relation is absent from codec: {row.relation_id}"
            )
        if row.target_kind == "entity":
            _entity_target(row.target)
        else:
            row.target.encode("utf-8")
        _validated_qualifiers(row)
        if not isinstance(row.provenance_id, str):
            raise ValueError("graph row provenance_id must be a string")
        row.provenance_id.encode("utf-8")
        materialized.append(row)

    ordered = tuple(sorted(materialized, key=lambda row: row.address))
    for previous, current in zip(ordered, ordered[1:]):
        if previous.address == current.address:
            raise ValueError(f"duplicate graph address: {current.address}")
    return ordered


def _append_blob(blob: bytearray, value: bytes) -> tuple[int, int]:
    offset = len(blob)
    blob.extend(value)
    if len(blob) > _UINT64_MAX:
        raise ValueError("packed graph blob exceeds uint64 address space")
    return offset, len(value)


def _encode_rows(
    rows: tuple[GraphRow, ...],
    relation_indices: Mapping[str, int],
) -> tuple[bytes, bytes]:
    headers = bytearray()
    blobs = bytearray()
    for row in rows:
        if row.target_kind == "entity":
            target_value = _entity_target(row.target)
            target_length = 0
        else:
            target_value, target_length = _append_blob(
                blobs,
                row.target.encode("utf-8"),
            )
        qualifier_offset, qualifier_length = _append_blob(
            blobs,
            _qualifier_bytes(_validated_qualifiers(row)),
        )
        provenance_offset, provenance_length = _append_blob(
            blobs,
            row.provenance_id.encode("utf-8"),
        )
        headers.extend(
            _ROW_STRUCT.pack(
                row.source_id,
                relation_indices[row.relation_id],
                _DIRECTION_TO_CODE[row.direction],
                _TARGET_KIND_TO_CODE[row.target_kind],
                target_value,
                target_length,
                qualifier_offset,
                qualifier_length,
                provenance_offset,
                provenance_length,
            )
        )
    return bytes(headers), bytes(blobs)


def _encode_index(
    rows: tuple[GraphRow, ...],
    relation_indices: Mapping[str, int],
    capacity: int,
) -> bytes:
    index = bytearray(_INDEX_STRUCT.pack(0, _UINT64_MAX) * capacity)

    mask = capacity - 1
    for row_index, row in enumerate(rows):
        direction_code = _DIRECTION_TO_CODE[row.direction]
        key = _key_bytes(
            row.source_id,
            relation_indices[row.relation_id],
            direction_code,
        )
        key_hash = _checked_hash(key)
        slot = key_hash & mask
        for _ in range(capacity):
            offset = slot * _INDEX_STRUCT.size
            _, stored_row_index = _INDEX_STRUCT.unpack_from(index, offset)
            if stored_row_index == _UINT64_MAX:
                _INDEX_STRUCT.pack_into(
                    index,
                    offset,
                    key_hash,
                    row_index,
                )
                break
            slot = (slot + 1) & mask
        else:
            raise ValueError("packed graph index has no empty slot")
    return bytes(index)


def _snapshot_sha256(rows: tuple[GraphRow, ...]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_row_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    try:
        value = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid packed graph manifest") from exc
    if not isinstance(value, dict) or set(value) != _MANIFEST_FIELDS:
        raise ValueError("invalid packed graph manifest fields")
    return value


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _validate_manifest(
    manifest: Mapping[str, Any],
    codec: RelationCodec,
) -> tuple[int, int]:
    if (
        isinstance(manifest["schema_version"], bool)
        or not isinstance(manifest["schema_version"], int)
        or manifest["schema_version"] != _SCHEMA_VERSION
    ):
        raise ValueError("unsupported packed graph schema version")
    if manifest["hash_algorithm"] != _HASH_ALGORITHM:
        raise ValueError("unsupported packed graph hash algorithm")
    if not _valid_sha256(manifest["codec_sha256"]):
        raise ValueError("invalid packed graph codec hash")
    if manifest["codec_sha256"] != codec.sha256():
        raise ValueError("relation codec hash mismatch")
    if not _valid_sha256(manifest["snapshot_sha256"]):
        raise ValueError("invalid packed graph snapshot hash")

    row_count = manifest["row_count"]
    capacity = manifest["capacity"]
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 0
    ):
        raise ValueError("packed graph row_count must be nonnegative")
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or capacity <= 0
        or capacity & (capacity - 1)
    ):
        raise ValueError("packed graph capacity must be a positive power of two")
    if row_count * 10 > capacity * 7:
        raise ValueError("packed graph index load factor exceeds 0.70")
    if capacity != _capacity_for_rows(row_count):
        raise ValueError("packed graph capacity is not canonical")

    file_hashes = manifest["file_sha256"]
    if not isinstance(file_hashes, dict) or set(file_hashes) != set(
        _FILE_NAMES
    ):
        raise ValueError("invalid packed graph file hash manifest")
    if any(not _valid_sha256(value) for value in file_hashes.values()):
        raise ValueError("invalid packed graph file hash")
    return row_count, capacity


def _verify_directory(
    path: Path,
    manifest: Mapping[str, Any],
    row_count: int,
    capacity: int,
) -> None:
    try:
        entries = {entry.name for entry in path.iterdir()}
    except OSError as exc:
        raise ValueError("invalid packed graph directory") from exc
    expected_entries = {"manifest.json", *_FILE_NAMES}
    if entries != expected_entries:
        raise ValueError("invalid packed graph directory entries")

    for name in _FILE_NAMES:
        file_path = path / name
        if file_path.is_symlink() or not file_path.is_file():
            raise ValueError(f"invalid packed graph file: {name}")
        if _sha256_file(file_path) != manifest["file_sha256"][name]:
            raise ValueError(f"file hash mismatch: {name}")

    expected_sizes = {
        "index.bin": capacity * _INDEX_STRUCT.size,
        "rows.bin": row_count * _ROW_STRUCT.size,
    }
    for name, expected_size in expected_sizes.items():
        try:
            actual_size = (path / name).stat().st_size
        except OSError as exc:
            raise ValueError(f"invalid packed graph file: {name}") from exc
        if actual_size != expected_size:
            raise ValueError(f"invalid {name} size")


def _map_file(path: Path) -> mmap.mmap | bytes:
    size = path.stat().st_size
    if size == 0:
        return b""
    with path.open("rb") as stream:
        return mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)


class PackedGraphStore:
    def __init__(
        self,
        path: Path,
        codec: RelationCodec,
        manifest: Mapping[str, Any],
        index_map: mmap.mmap | bytes,
        row_map: mmap.mmap | bytes,
        blob_map: mmap.mmap | bytes,
    ) -> None:
        self.path = path
        self._codec = codec
        self._relation_indices = {
            relation_id: index
            for index, relation_id in enumerate(codec.relation_ids)
        }
        self._row_count = manifest["row_count"]
        self._capacity = manifest["capacity"]
        self._snapshot_hash = manifest["snapshot_sha256"]
        self._index_map = index_map
        self._row_map = row_map
        self._blob_map = blob_map
        self._closed = False
        self._provenance_ranges: (
            dict[str, tuple[tuple[int, int], ...]] | None
        ) = None
        self._max_entity_id: int | None = None
        self.hits = 0
        self.misses = 0
        self._stats = StoreStats(
            rows=self._row_count,
            index_bytes=len(index_map),
            row_bytes=len(row_map),
            blob_bytes=len(blob_map),
        )

    @classmethod
    def build(
        cls,
        path: str | Path,
        rows: Iterable[GraphRow],
        codec: RelationCodec,
    ) -> PackedGraphStore:
        destination = Path(path)
        if os.path.lexists(destination):
            raise FileExistsError(f"packed graph path already exists: {path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        relation_indices = {
            relation_id: index
            for index, relation_id in enumerate(codec.relation_ids)
        }
        ordered_rows = _materialize_rows(rows, relation_indices)
        capacity = _capacity_for_rows(len(ordered_rows))
        row_bytes, blob_bytes = _encode_rows(
            ordered_rows,
            relation_indices,
        )
        index_bytes = _encode_index(
            ordered_rows,
            relation_indices,
            capacity,
        )

        temporary = Path(
            tempfile.mkdtemp(
                dir=destination.parent,
                prefix=f".{destination.name}.tmp-",
            )
        )
        try:
            contents = {
                "index.bin": index_bytes,
                "rows.bin": row_bytes,
                "blobs.bin": blob_bytes,
            }
            for name in _FILE_NAMES:
                _write_file(temporary / name, contents[name])
            manifest = {
                "schema_version": _SCHEMA_VERSION,
                "hash_algorithm": _HASH_ALGORITHM,
                "codec_sha256": codec.sha256(),
                "row_count": len(ordered_rows),
                "capacity": capacity,
                "snapshot_sha256": _snapshot_sha256(ordered_rows),
                "file_sha256": {
                    name: hashlib.sha256(contents[name]).hexdigest()
                    for name in _FILE_NAMES
                },
            }
            _write_file(
                temporary / "manifest.json",
                _canonical_json_bytes(manifest),
            )
            verified = cls._load(
                temporary,
                codec,
            )
            verified.close()
            if os.path.lexists(destination):
                raise FileExistsError(
                    f"packed graph path already exists: {path}"
                )
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        try:
            return cls.load(destination, codec)
        except BaseException:
            shutil.rmtree(destination)
            raise

    @classmethod
    def load(
        cls,
        path: str | Path,
        codec: RelationCodec,
    ) -> PackedGraphStore:
        return cls._load(Path(path), codec)

    @classmethod
    def _load(
        cls,
        path: Path,
        codec: RelationCodec,
    ) -> PackedGraphStore:
        if path.is_symlink() or not path.is_dir():
            raise ValueError("packed graph path must be a directory")
        manifest = _read_manifest(path)
        row_count, capacity = _validate_manifest(manifest, codec)
        _verify_directory(path, manifest, row_count, capacity)

        mappings: list[mmap.mmap | bytes] = []
        try:
            for name in _FILE_NAMES:
                mappings.append(_map_file(path / name))
            store = cls(
                path,
                codec,
                manifest,
                mappings[0],
                mappings[1],
                mappings[2],
            )
            store._validate_content()
            return store
        except BaseException:
            for mapping in mappings:
                close = getattr(mapping, "close", None)
                if close is not None:
                    close()
            raise

    def _header(self, row_index: int) -> tuple[int, ...]:
        if not 0 <= row_index < self._row_count:
            raise ValueError("packed graph row index is out of bounds")
        return _ROW_STRUCT.unpack_from(
            self._row_map,
            row_index * _ROW_STRUCT.size,
        )

    def _blob(self, offset: int, length: int, name: str) -> bytes:
        end = offset + length
        if end < offset or end > len(self._blob_map):
            raise ValueError(f"packed graph {name} blob is out of bounds")
        return bytes(self._blob_map[offset:end])

    def _decode_qualifiers(
        self,
        offset: int,
        length: int,
    ) -> tuple[tuple[str, str], ...]:
        raw = self._blob(offset, length, "qualifier")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid packed graph qualifier JSON") from exc
        if not isinstance(value, list) or any(
            not isinstance(qualifier, list)
            or len(qualifier) != 2
            or not all(isinstance(item, str) for item in qualifier)
            for qualifier in value
        ):
            raise ValueError("invalid packed graph qualifiers")
        qualifiers = tuple((item[0], item[1]) for item in value)
        if _qualifier_bytes(qualifiers) != raw:
            raise ValueError("packed graph qualifier JSON is not canonical")
        return qualifiers

    def _decode_row(self, row_index: int) -> GraphRow:
        (
            source_id,
            relation_index,
            direction_code,
            target_kind_code,
            target_value,
            target_length,
            qualifier_offset,
            qualifier_length,
            provenance_offset,
            provenance_length,
        ) = self._header(row_index)
        if relation_index >= len(self._codec.relation_ids):
            raise ValueError("packed graph relation index is out of bounds")
        try:
            direction = _CODE_TO_DIRECTION[direction_code]
        except KeyError as exc:
            raise ValueError("invalid packed graph direction code") from exc
        try:
            target_kind = _CODE_TO_TARGET_KIND[target_kind_code]
        except KeyError as exc:
            raise ValueError("invalid packed graph target kind code") from exc

        if target_kind == "entity":
            if target_length != 0:
                raise ValueError("invalid packed entity target header")
            target = str(target_value)
        else:
            raw_target = self._blob(target_value, target_length, "target")
            try:
                target = raw_target.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("invalid packed graph target UTF-8") from exc
            if not target:
                raise ValueError("packed graph literal target must be nonempty")

        qualifiers = self._decode_qualifiers(
            qualifier_offset,
            qualifier_length,
        )
        try:
            provenance_id = self._blob(
                provenance_offset,
                provenance_length,
                "provenance",
            ).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid packed graph provenance UTF-8") from exc
        return GraphRow(
            source_id=source_id,
            relation_id=self._codec.relation_ids[relation_index],
            direction=direction,
            target_kind=target_kind,
            target=target,
            qualifiers=qualifiers,
            provenance_id=provenance_id,
        )

    def _validate_content(self) -> None:
        expected_blob_offset = 0
        previous_address: GraphAddress | None = None
        snapshot = hashlib.sha256()
        for row_index in range(self._row_count):
            header = self._header(row_index)
            target_kind_code = header[3]
            if target_kind_code == _TARGET_KIND_TO_CODE["literal"]:
                if header[4] != expected_blob_offset:
                    raise ValueError("invalid packed graph target blob offset")
                expected_blob_offset += header[5]
            elif target_kind_code == _TARGET_KIND_TO_CODE["entity"]:
                if header[5] != 0:
                    raise ValueError("invalid packed entity target header")
            if header[6] != expected_blob_offset:
                raise ValueError("invalid packed graph qualifier blob offset")
            expected_blob_offset += header[7]
            if header[8] != expected_blob_offset:
                raise ValueError("invalid packed graph provenance blob offset")
            expected_blob_offset += header[9]

            row = self._decode_row(row_index)
            if previous_address is not None:
                if row.address == previous_address:
                    raise ValueError(
                        f"duplicate graph address: {row.address}"
                    )
                if row.address < previous_address:
                    raise ValueError("packed graph rows are not sorted")
            previous_address = row.address
            snapshot.update(_canonical_row_bytes(row))
            snapshot.update(b"\n")
        if expected_blob_offset != len(self._blob_map):
            raise ValueError("invalid packed graph blob size")
        if snapshot.hexdigest() != self._snapshot_hash:
            raise ValueError("packed graph snapshot hash mismatch")

        seen_rows = bytearray(self._row_count)
        occupied = 0
        mask = self._capacity - 1
        for slot in range(self._capacity):
            key_hash, row_index = _INDEX_STRUCT.unpack_from(
                self._index_map,
                slot * _INDEX_STRUCT.size,
            )
            if row_index == _UINT64_MAX:
                if key_hash != 0:
                    raise ValueError("invalid packed graph empty index slot")
                continue
            if row_index >= self._row_count:
                raise ValueError("packed graph index row is out of bounds")
            if seen_rows[row_index]:
                raise ValueError("packed graph index contains a duplicate row")
            row = self._decode_row(row_index)
            relation_index = self._relation_indices[row.relation_id]
            expected_hash = _checked_hash(
                _key_bytes(
                    row.source_id,
                    relation_index,
                    _DIRECTION_TO_CODE[row.direction],
                ),
            )
            if key_hash != expected_hash:
                raise ValueError("packed graph index hash mismatch")

            probe = key_hash & mask
            for _ in range(self._capacity):
                if probe == slot:
                    break
                _, probe_row = _INDEX_STRUCT.unpack_from(
                    self._index_map,
                    probe * _INDEX_STRUCT.size,
                )
                if probe_row == _UINT64_MAX:
                    raise ValueError("invalid packed graph probe chain")
                probe = (probe + 1) & mask
            else:
                raise ValueError("invalid packed graph probe chain")
            seen_rows[row_index] = 1
            occupied += 1
        if occupied != self._row_count or any(value == 0 for value in seen_rows):
            raise ValueError("packed graph index does not cover every row")

    def lookup(self, address: GraphAddress) -> GraphRow | None:
        if self._closed:
            raise ValueError("packed graph store is closed")
        relation_index = self._relation_indices.get(address.relation_id)
        if (
            relation_index is None
            or address.source_id > _UINT64_MAX
            or address.source_id < 0
        ):
            self.misses += 1
            return None
        direction_code = _DIRECTION_TO_CODE.get(address.direction)
        if direction_code is None:
            self.misses += 1
            return None
        key_hash = _checked_hash(
            _key_bytes(
                address.source_id,
                relation_index,
                direction_code,
            ),
        )
        mask = self._capacity - 1
        slot = key_hash & mask
        for _ in range(self._capacity):
            stored_hash, row_index = _INDEX_STRUCT.unpack_from(
                self._index_map,
                slot * _INDEX_STRUCT.size,
            )
            if row_index == _UINT64_MAX:
                self.misses += 1
                return None
            if stored_hash == key_hash:
                row = self._decode_row(row_index)
                if row.address == address:
                    self.hits += 1
                    return row
            slot = (slot + 1) & mask
        self.misses += 1
        return None

    def rows_for_provenance(self, provenance_id: str) -> tuple[GraphRow, ...]:
        if self._closed:
            raise ValueError("packed graph store is closed")
        if not isinstance(provenance_id, str) or not provenance_id:
            raise ValueError("provenance_id must be a nonempty string")
        if self._provenance_ranges is None:
            mutable: dict[str, list[tuple[int, int]]] = {}
            previous: str | None = None
            start = 0
            for row_index in range(self._row_count):
                header = self._header(row_index)
                current = self._blob(
                    header[8],
                    header[9],
                    "provenance",
                ).decode("utf-8")
                if previous is None:
                    previous = current
                    start = row_index
                elif current != previous:
                    mutable.setdefault(previous, []).append(
                        (start, row_index)
                    )
                    previous = current
                    start = row_index
            if previous is not None:
                mutable.setdefault(previous, []).append(
                    (start, self._row_count)
                )
            self._provenance_ranges = {
                key: tuple(ranges) for key, ranges in mutable.items()
            }
        return tuple(
            self._decode_row(row_index)
            for start, end in self._provenance_ranges.get(
                provenance_id,
                (),
            )
            for row_index in range(start, end)
        )

    def max_entity_id(self) -> int:
        if self._closed:
            raise ValueError("packed graph store is closed")
        if self._max_entity_id is None:
            maximum: int | None = None
            for row_index in range(self._row_count):
                header = self._header(row_index)
                maximum = (
                    header[0]
                    if maximum is None
                    else max(maximum, header[0])
                )
                if header[3] == _TARGET_KIND_TO_CODE["entity"]:
                    maximum = max(maximum, header[4])
            if maximum is None:
                raise ValueError("empty graph store has no entity id")
            self._max_entity_id = maximum
        return self._max_entity_id

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return self._row_count

    def snapshot_sha256(self) -> str:
        return self._snapshot_hash

    def stats(self) -> StoreStats:
        return self._stats

    def close(self) -> None:
        if self._closed:
            return
        for mapping in (self._index_map, self._row_map, self._blob_map):
            close = getattr(mapping, "close", None)
            if close is not None:
                close()
        self._closed = True

    def __enter__(self) -> PackedGraphStore:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
