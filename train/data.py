"""Exact packed-sequence loading over token, mask, and weight shards.

Legacy corpora may pair the flat uint16 token stream with a binary uint8 loss
mask (1 = loss ON). Relational corpora instead use per-arm target-weight
sidecars and normally omit the legacy mask. One or more files form one logical,
cyclic stream. Every batch reads ``sequences * ctx + 1`` contiguous tokens, so
each causal target occurs once without row-boundary gaps or duplicates.

The monotonic global cursor is checkpointed. Rank plans partition each optimizer
update without mutating it; every rank advances it by the same global target
count only after the update completes.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np
import torch


PARALLEL_SIDECAR_V2_CONTRACT = {
    "format": "memorysplit-parallel-corpus-v2",
    "receipt_field": "sidecar_sets",
    "artifact_fields": ("bytes", "path", "sha256"),
    "set_fields": (
        "artifacts",
        "dtype",
        "items",
        "name",
        "stream_sha256",
    ),
    "defined_names": ("loss_mask", "target_weights"),
    "alignment": {
        "items": "packed_tokens",
        "shard_order": "token_assignments",
    },
    "padding_rule": {
        "sidecar": "target_weights",
        "required_value": 0,
    },
}
"""Loader-side integration contract for a future sidecar-aware publication.

The publisher must put named sidecar sets in the canonical receipt. Each set
must bind an ordered artifact list (path, byte length, SHA-256), scalar dtype,
logical item count, and whole-stream SHA-256. A padded token publication must
bind ``target_weights`` of equal packed length and require zero at every
padding target. Version 1 has no such namespace and remains unsupported for
Split90 sidecars.
"""


@dataclass(frozen=True)
class BatchSlice:
    global_start: int
    sequence_count: int
    ctx: int

    def __post_init__(self) -> None:
        if type(self.global_start) is not int or self.global_start < 0:
            raise ValueError("BatchSlice global_start must be a non-negative integer")
        if type(self.sequence_count) is not int or self.sequence_count <= 0:
            raise ValueError("BatchSlice sequence_count must be a positive integer")
        if type(self.ctx) is not int or self.ctx <= 0:
            raise ValueError("BatchSlice ctx must be a positive integer")

    @property
    def target_count(self) -> int:
        return self.sequence_count * self.ctx


def rank_sequence_counts(total_sequences: int, world_size: int) -> tuple[int, ...]:
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
    ):
        raise ValueError("world_size must be positive")
    if (
        isinstance(total_sequences, bool)
        or not isinstance(total_sequences, int)
        or total_sequences <= 0
    ):
        raise ValueError("total_sequences must be positive")
    base, extra = divmod(total_sequences, world_size)
    return tuple(base + (rank < extra) for rank in range(world_size))


def rank_batch_plan(
    *,
    global_cursor: int,
    total_sequences: int,
    ctx: int,
    micro_batch_size: int,
    rank: int,
    world_size: int,
) -> tuple[BatchSlice, ...]:
    for value, name in (
        (global_cursor, "global_cursor"),
        (ctx, "ctx"),
        (micro_batch_size, "micro_batch_size"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if name == "global_cursor" else 1)
        ):
            qualifier = "non-negative" if name == "global_cursor" else "positive"
            raise ValueError(f"{name} must be {qualifier}")
    counts = rank_sequence_counts(total_sequences, world_size)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    local_sequences = counts[rank]
    rank_sequence_start = sum(counts[:rank])
    batches = []
    consumed = 0
    while consumed < local_sequences:
        sequence_count = min(micro_batch_size, local_sequences - consumed)
        batches.append(
            BatchSlice(
                global_start=global_cursor
                + (rank_sequence_start + consumed) * ctx,
                sequence_count=sequence_count,
                ctx=ctx,
            )
        )
        consumed += sequence_count
    return tuple(batches)


def synchronized_rank_batch_plan(
    *,
    global_cursor: int,
    total_sequences: int,
    ctx: int,
    micro_batch_size: int,
    rank: int,
    world_size: int,
) -> tuple[BatchSlice | None, ...]:
    """Return a rank plan padded to a common DDP backward-call count."""

    plan = rank_batch_plan(
        global_cursor=global_cursor,
        total_sequences=total_sequences,
        ctx=ctx,
        micro_batch_size=micro_batch_size,
        rank=rank,
        world_size=world_size,
    )
    max_local_sequences = max(rank_sequence_counts(total_sequences, world_size))
    microsteps = (max_local_sequences + micro_batch_size - 1) // micro_batch_size
    return (*plan, *((None,) * (microsteps - len(plan))))


def _open_directory_path(path: Path) -> int:
    absolute = path.absolute()
    parts = absolute.parts
    current = os.open(
        parts[0],
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        for component in parts[1:]:
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current,
                )
            except OSError as error:
                raise ValueError(
                    f"directory path has a missing, symlinked, or unsafe component: {path}"
                ) from error
            os.close(current)
            current = next_fd
        return current
    except BaseException:
        os.close(current)
        raise


def _open_parent_directory(path: Path) -> tuple[int, str]:
    absolute = path.absolute()
    if absolute.name in {"", ".", ".."}:
        raise ValueError(f"file path is invalid: {path}")
    return _open_directory_path(absolute.parent), absolute.name


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(fd, 1 << 20, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


class _PinnedFile:
    def __init__(
        self,
        fd: int,
        *,
        path: Path,
        label: str,
        item_bytes: int | None = None,
    ) -> None:
        self.path = path
        self.label = label
        self.handle = os.fdopen(fd, "rb", closefd=True)
        try:
            metadata = os.fstat(self.handle.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"{label} file is not regular or safe: {path}")
            self.byte_count = metadata.st_size
            if item_bytes is not None and (
                self.byte_count <= 0 or self.byte_count % item_bytes
            ):
                raise ValueError(f"{label} file byte length is invalid: {path}")
            self.item_count = (
                self.byte_count // item_bytes if item_bytes is not None else None
            )
            self.sha256 = _sha256_fd(self.handle.fileno())
        except BaseException:
            self.handle.close()
            raise


    @classmethod
    def open_path(
        cls,
        path: Path,
        *,
        label: str,
        item_bytes: int | None = None,
    ) -> _PinnedFile:
        parent_fd, name = _open_parent_directory(path)
        try:
            return cls.open_at(
                parent_fd,
                name,
                path=path,
                label=label,
                item_bytes=item_bytes,
            )
        finally:
            os.close(parent_fd)

    @classmethod
    def open_at(
        cls,
        parent_fd: int,
        name: str,
        *,
        path: Path,
        label: str,
        item_bytes: int | None = None,
    ) -> _PinnedFile:
        if "/" in name or name in {"", ".", ".."}:
            raise ValueError(f"{label} path is unsafe: {path}")
        try:
            fd = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise ValueError(f"{label} file is missing, symlinked, or unsafe: {path}") from error
        try:
            return cls(fd, path=path, label=label, item_bytes=item_bytes)
        except BaseException:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise

    def read_bytes(self) -> bytes:
        chunks = []
        offset = 0
        while offset < self.byte_count:
            chunk = os.pread(
                self.handle.fileno(),
                min(1 << 20, self.byte_count - offset),
                offset,
            )
            if not chunk:
                raise ValueError(f"{self.label} changed length while pinned: {self.path}")
            chunks.append(chunk)
            offset += len(chunk)
        return b"".join(chunks)

    def memmap(self, dtype) -> np.memmap:
        return np.memmap(self.handle, dtype=dtype, mode="r")


def _content_provenance(
    files: tuple[_PinnedFile, ...] | None,
    *,
    item_bytes: int,
) -> dict[str, int | str] | None:
    if files is None:
        return None
    if len(files) == 1:
        sha256 = files[0].sha256
    else:
        digest = hashlib.sha256()
        for pinned in files:
            offset = 0
            while offset < pinned.byte_count:
                chunk = os.pread(
                    pinned.handle.fileno(),
                    min(1 << 20, pinned.byte_count - offset),
                    offset,
                )
                if not chunk:
                    raise ValueError(f"pinned file changed length: {pinned.path}")
                digest.update(chunk)
                offset += len(chunk)
        sha256 = digest.hexdigest()
    return {
        "bytes": sum(pinned.byte_count for pinned in files),
        "items": sum(pinned.byte_count // item_bytes for pinned in files),
        "sha256": sha256,
    }


class PackedShards:
    def __init__(
        self,
        bin_path: str | Path,
        mask_path: str | Path | None,
        ctx: int,
        batch_size: int,
        device: str = "cpu",
        start_cursor: int = 0,
        seed: int = 0,
        weights_path: str | Path | None = None,
    ):
        if not isinstance(bin_path, (str, Path)):
            raise ValueError(
                "multiple token files require a verified parallel corpus publication"
            )
        token_paths = (Path(bin_path),)
        mask_paths = (Path(mask_path),) if mask_path is not None else None
        weight_paths = (Path(weights_path),) if weights_path is not None else None
        opened: list[_PinnedFile] = []
        try:
            token_files = (
                _PinnedFile.open_path(
                    token_paths[0],
                    label="token",
                    item_bytes=np.dtype(np.uint16).itemsize,
                ),
            )
            opened.extend(token_files)
            token_count = token_files[0].item_count
            assert token_count is not None
            mask_files = None
            if mask_paths is not None:
                mask_files = (
                    _PinnedFile.open_path(
                        mask_paths[0],
                        label="mask",
                        item_bytes=np.dtype(np.uint8).itemsize,
                    ),
                )
                opened.extend(mask_files)
                if mask_files[0].item_count != token_count:
                    raise ValueError("mask sidecar length does not match token length")
            weight_files = None
            if weight_paths is not None:
                weight_files = (
                    _PinnedFile.open_path(
                        weight_paths[0],
                        label="weights",
                        item_bytes=np.dtype(np.uint8).itemsize,
                    ),
                )
                opened.extend(weight_files)
                if weight_files[0].item_count != token_count:
                    raise ValueError("weights sidecar length does not match token length")
            provenance = {
                "format_version": 3,
                "kind": "legacy-single-file",
                "token_count": token_count,
                "tokens": _content_provenance(token_files, item_bytes=2),
                "mask": _content_provenance(mask_files, item_bytes=1),
                "weights": _content_provenance(weight_files, item_bytes=1),
            }
            self._initialize(
                token_files=token_files,
                mask_files=mask_files,
                weight_files=weight_files,
                ctx=ctx,
                batch_size=batch_size,
                device=device,
                start_cursor=start_cursor,
                provenance=provenance,
                publication_update_tokens=None,
                directory_fds=(),
                open_files=tuple(opened),
            )
        except BaseException:
            for pinned in opened:
                pinned.handle.close()
            raise

    @classmethod
    def from_parallel_corpus(
        cls,
        root: str | Path,
        *,
        ctx: int,
        batch_size: int,
        device: str = "cpu",
        start_cursor: int = 0,
        seed: int = 0,
        mask_path: str | Path | None = None,
        weights_path: str | Path | None = None,
    ) -> PackedShards:
        """Open a fully verified immutable parallel-corpus publication.

        The v1 receipt binds token shards but has no sidecar namespace. Masks
        and target weights are therefore rejected instead of being paired by
        filename convention.
        """

        del seed
        if mask_path is not None or weights_path is not None:
            raise ValueError(
                "parallel corpus v1 does not bind sidecars; "
                "mask_path and weights_path are unsupported"
            )
        from corpusgen.parallel import assignments_from_bytes, verify_parallel_corpus
        from corpusgen.parallel.canonical import canonical_json_bytes

        publication = Path(root)
        receipt = verify_parallel_corpus(publication)
        if receipt["padding_tokens"] != 0:
            raise ValueError(
                "parallel corpus padding has no bound zero-weight sidecar"
            )
        directory_fds: list[int] = []
        opened: list[_PinnedFile] = []
        try:
            root_fd = _open_directory_path(publication)
            directory_fds.append(root_fd)
            try:
                shards_fd = os.open(
                    "shards",
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=root_fd,
                )
            except OSError as error:
                raise ValueError(
                    "parallel corpus shards directory is missing, symlinked, or unsafe"
                ) from error
            directory_fds.append(shards_fd)

            receipt_file = _PinnedFile.open_at(
                root_fd,
                "receipt.json",
                path=publication / "receipt.json",
                label="parallel corpus receipt",
            )
            opened.append(receipt_file)
            receipt_bytes = receipt_file.read_bytes()
            try:
                pinned_receipt = json.loads(receipt_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("parallel corpus receipt is invalid JSON") from error
            if (
                pinned_receipt != receipt
                or canonical_json_bytes(pinned_receipt) != receipt_bytes
            ):
                raise ValueError(
                    "parallel corpus receipt changed after semantic verification"
                )

            artifacts: dict[str, _PinnedFile] = {}
            for artifact in receipt["artifacts"]:
                relative = Path(artifact["path"])
                if len(relative.parts) == 1:
                    parent_fd = root_fd
                    name = relative.parts[0]
                elif len(relative.parts) == 2 and relative.parts[0] == "shards":
                    parent_fd = shards_fd
                    name = relative.parts[1]
                else:
                    raise ValueError(
                        f"parallel corpus artifact path is unsafe: {artifact['path']}"
                    )
                pinned = _PinnedFile.open_at(
                    parent_fd,
                    name,
                    path=publication / relative,
                    label="parallel corpus artifact",
                )
                opened.append(pinned)
                if (
                    pinned.byte_count != artifact["bytes"]
                    or pinned.sha256 != artifact["sha256"]
                ):
                    raise ValueError(
                        f"publication artifact digest drift: {artifact['path']}"
                    )
                artifacts[artifact["path"]] = pinned

            assignments = assignments_from_bytes(
                artifacts["assignments.jsonl"].read_bytes()
            )
            token_paths = tuple(
                publication / "shards" / f"{assignment.shard_id}.bin"
                for assignment in assignments
            )
            token_files = tuple(
                artifacts[f"shards/{assignment.shard_id}.bin"]
                for assignment in assignments
            )
            for pinned, assignment in zip(
                token_files,
                assignments,
                strict=True,
            ):
                expected = assignment.token_end - assignment.token_start
                if pinned.byte_count <= 0 or pinned.byte_count % 2:
                    raise ValueError(
                        f"token shard byte length is invalid: {pinned.path}"
                    )
                if pinned.byte_count // 2 != expected:
                    raise ValueError(
                        f"token shard length does not match assignment: {pinned.path}"
                    )
            provenance = {
                "format_version": 3,
                "kind": "parallel-publication",
                "token_count": receipt["packed_tokens"],
                "receipt_sha256": receipt_file.sha256,
                "build_id": receipt["build_id"],
                "ordered_stream_sha256": receipt["ordered_stream_sha256"],
                "packed_stream_sha256": receipt["packed_stream_sha256"],
                "mask": None,
                "weights": None,
            }
            result = cls.__new__(cls)
            result._initialize(
                token_files=token_files,
                mask_files=None,
                weight_files=None,
                ctx=ctx,
                batch_size=batch_size,
                device=device,
                start_cursor=start_cursor,
                provenance=provenance,
                publication_update_tokens=receipt["config"]["update_tokens"],
                directory_fds=tuple(directory_fds),
                open_files=tuple(opened),
            )
            return result
        except BaseException:
            for pinned in opened:
                pinned.handle.close()
            for fd in directory_fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise

    def _initialize(
        self,
        *,
        token_files: tuple[_PinnedFile, ...],
        mask_files: tuple[_PinnedFile, ...] | None,
        weight_files: tuple[_PinnedFile, ...] | None,
        ctx: int,
        batch_size: int,
        device: str,
        start_cursor: int,
        provenance: dict,
        publication_update_tokens: int | None,
        directory_fds: tuple[int, ...],
        open_files: tuple[_PinnedFile, ...],
    ) -> None:
        for value, name in ((ctx, "ctx"), (batch_size, "batch_size")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(start_cursor, bool)
            or not isinstance(start_cursor, int)
            or start_cursor < 0
        ):
            raise ValueError("start_cursor must be a non-negative integer")
        self._directory_fds = directory_fds
        self._open_files = open_files
        self.token_paths = tuple(pinned.path for pinned in token_files)
        self._token_shards = tuple(
            pinned.memmap(np.uint16) for pinned in token_files
        )
        self._shard_lengths = tuple(len(shard) for shard in self._token_shards)
        self._shard_ends = np.cumsum(self._shard_lengths).tolist()
        self.tokens = (
            self._token_shards[0]
            if len(self._token_shards) == 1
            else self._token_shards
        )

        self.mask_paths = (
            tuple(pinned.path for pinned in mask_files)
            if mask_files is not None
            else None
        )
        if mask_files is not None:
            self._mask_shards = tuple(
                pinned.memmap(np.uint8) for pinned in mask_files
            )
            self.mask = (
                self._mask_shards[0]
                if len(self._mask_shards) == 1
                else self._mask_shards
            )
        else:
            self._mask_shards = None
            self.mask = None

        self.weight_paths = (
            tuple(pinned.path for pinned in weight_files)
            if weight_files is not None
            else None
        )
        if weight_files is not None:
            self._weight_shards = tuple(
                pinned.memmap(np.uint8) for pinned in weight_files
            )
            self.target_weights = (
                self._weight_shards[0]
                if len(self._weight_shards) == 1
                else self._weight_shards
            )
        else:
            self._weight_shards = None
            self.target_weights = None
        self.ctx = ctx
        self.batch_size = batch_size
        self.device = device
        self.n_tokens = sum(self._shard_lengths)
        if self.n_tokens <= 0:
            raise ValueError("token corpus must not be empty")
        self.provenance = provenance
        self.publication_update_tokens = publication_update_tokens
        self.targets_per_update: int | None = None
        self.global_cursor = start_cursor
        self.cursor = start_cursor % self.n_tokens
        self.epoch = start_cursor // self.n_tokens
        self._last_batch_start: int | None = None
        if self.n_tokens < self.ctx:
            raise ValueError("corpus smaller than one sequence")

    def validate_update_alignment(self, targets_per_update: int) -> None:
        if (
            isinstance(targets_per_update, bool)
            or not isinstance(targets_per_update, int)
            or targets_per_update <= 0
        ):
            raise ValueError("targets_per_update must be a positive integer")
        if self.publication_update_tokens is not None and (
            targets_per_update != self.publication_update_tokens
        ):
            raise ValueError(
                "training targets_per_update does not match publication update_tokens"
            )
        if self.n_tokens < targets_per_update:
            raise ValueError("corpus is smaller than one optimizer update")
        if self.n_tokens % targets_per_update:
            raise ValueError("corpus token count is not update-aligned")
        if len(self._token_shards) > 1 and any(
            length % targets_per_update for length in self._shard_lengths
        ):
            raise ValueError("token shards are not update-aligned")
        if self.global_cursor % targets_per_update:
            raise ValueError("global cursor is not update-aligned")
        self.targets_per_update = targets_per_update

    def _read_shards(
        self,
        shards: tuple[np.memmap, ...],
        start: int,
        length: int,
    ) -> np.ndarray:
        if start < 0 or length < 0:
            raise ValueError("stream offsets and lengths must be non-negative")
        result = np.empty(length, dtype=shards[0].dtype)
        output_offset = 0
        position = start % self.n_tokens
        while output_offset < length:
            shard_index = bisect_right(self._shard_ends, position)
            shard_start = 0 if shard_index == 0 else self._shard_ends[shard_index - 1]
            local_start = position - shard_start
            take = min(
                length - output_offset,
                self._shard_lengths[shard_index] - local_start,
            )
            result[output_offset : output_offset + take] = shards[shard_index][
                local_start : local_start + take
            ]
            output_offset += take
            position = (position + take) % self.n_tokens
        return result

    def _window(self, start: int, length: int) -> tuple[np.ndarray, np.ndarray | None]:
        toks = self._read_shards(self._token_shards, start, length)
        msk = (
            self._read_shards(self._mask_shards, start, length)
            if self._mask_shards is not None
            else None
        )
        return toks, msk

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.device.startswith("cuda"):
            return tensor.pin_memory().to(self.device, non_blocking=True)
        if self.device != "cpu":
            return tensor.to(self.device)
        return tensor

    def batch_from_slice(
        self,
        batch_slice: BatchSlice,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_slice.ctx != self.ctx:
            raise ValueError("batch slice context does not match dataset context")
        target_count = batch_slice.target_count
        toks, msk = self._window(batch_slice.global_start, target_count + 1)
        toks = toks.astype(np.int64)
        shape = (batch_slice.sequence_count, self.ctx)
        x = torch.from_numpy(toks[:-1].reshape(shape).copy())
        y = torch.from_numpy(toks[1:].reshape(shape).copy())
        if msk is not None:
            m = msk[1:].reshape(shape)
            y[torch.from_numpy((m == 0).copy())] = -100
        return self._to_device(x), self._to_device(y)

    def weighted_batch_from_slice(
        self,
        batch_slice: BatchSlice,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, targets = self.batch_from_slice(batch_slice)
        if self._weight_shards is None:
            weights = torch.ones_like(targets, dtype=torch.float32)
        else:
            raw = self._read_shards(
                self._weight_shards,
                batch_slice.global_start + 1,
                batch_slice.target_count,
            )
            weights = torch.from_numpy(
                raw.astype(np.float32).reshape(tuple(targets.shape)).copy()
            )
            weights = self._to_device(weights)
        return x, targets, weights

    def advance(self, target_count: int) -> None:
        if (
            isinstance(target_count, bool)
            or not isinstance(target_count, int)
            or target_count <= 0
        ):
            raise ValueError("target_count must be a positive integer")
        if (
            self.targets_per_update is not None
            and target_count != self.targets_per_update
        ):
            raise ValueError(
                "validated global cursor must advance by exactly targets_per_update"
            )
        self.global_cursor += target_count
        self.cursor = self.global_cursor % self.n_tokens
        self.epoch = self.global_cursor // self.n_tokens

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        target_count = self.batch_size * self.ctx
        start = self.global_cursor
        batch_slice = BatchSlice(start, self.batch_size, self.ctx)
        batch = self.batch_from_slice(batch_slice)
        self._last_batch_start = start
        self.advance(target_count)
        return batch

    def _aligned_next_token_weights_for_last_batch(self) -> torch.Tensor:
        assert self.target_weights is not None
        assert self._last_batch_start is not None
        span = self.batch_size * self.ctx + 1
        assert self._weight_shards is not None
        raw = self._read_shards(self._weight_shards, self._last_batch_start, span)
        raw = raw[1:].reshape(self.batch_size, self.ctx)
        weights = torch.from_numpy(raw.astype(np.float32, copy=True))
        return self._to_device(weights)

    def next_weighted_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, targets = self.next_batch()
        if self.target_weights is None:
            weights = torch.ones_like(targets, dtype=torch.float32)
        else:
            weights = self._aligned_next_token_weights_for_last_batch()
        return x, targets, weights

    def masked_value_batch(self, max_batches: int = 8) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Probe positions excluded by an optional legacy binary loss mask.

        Returns (x, y) where y is -100 everywhere EXCEPT masked-value targets —
        the complement of the training labels — sampled from the shard head.
        Returns None for target-weight-only relational corpora and for legacy
        corpora with no masked positions.
        """
        if self.mask is None:
            return None
        target_count = self.batch_size * self.ctx * max_batches
        toks, msk = self._window(0, target_count + 1)
        target_mask = msk[1:]
        if (target_mask == 0).sum() == 0:
            return None
        shape = (-1, self.ctx)
        toks = toks.astype(np.int64)
        x = torch.from_numpy(toks[:-1].reshape(shape).copy())
        y = torch.from_numpy(toks[1:].reshape(shape).copy())
        keep = torch.from_numpy((target_mask.reshape(shape) == 0).copy())
        y[~keep] = -100
        rows = keep.any(dim=1)
        if not rows.any():
            return None
        return x[rows], y[rows]

    def close(self) -> None:
        for shard_group_name in (
            "_token_shards",
            "_mask_shards",
            "_weight_shards",
        ):
            shard_group = getattr(self, shard_group_name, None) or ()
            for shard in shard_group:
                mapping = getattr(shard, "_mmap", None)
                if mapping is not None:
                    mapping.close()
            setattr(self, shard_group_name, None)
        for pinned in getattr(self, "_open_files", ()):
            if not pinned.handle.closed:
                pinned.handle.close()
        self._open_files = ()
        for fd in getattr(self, "_directory_fds", ()):
            try:
                os.close(fd)
            except OSError:
                pass
        self._directory_fds = ()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    def state_dict(self) -> dict:
        return {
            "format_version": 2,
            "global_cursor": self.global_cursor,
            "cursor": self.cursor,
            "epoch": self.epoch,
            "provenance": self.provenance,
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise ValueError("data state must be a dictionary")
        if state.get("format_version") != 2:
            raise ValueError("data state version is incompatible")
        expected_fields = {
            "format_version",
            "global_cursor",
            "cursor",
            "epoch",
            "provenance",
        }
        if set(state) != expected_fields:
            raise ValueError("data state fields do not match the contract")
        saved_provenance = state.get("provenance")
        if saved_provenance != self.provenance:
            raise ValueError("data shard provenance does not match checkpoint")
        global_cursor = state.get("global_cursor")
        if (
            isinstance(global_cursor, bool)
            or not isinstance(global_cursor, int)
            or global_cursor < 0
        ):
            raise ValueError("checkpoint global cursor is invalid")
        if state.get("cursor") != global_cursor % self.n_tokens:
            raise ValueError("checkpoint local cursor is inconsistent")
        if state.get("epoch") != global_cursor // self.n_tokens:
            raise ValueError("checkpoint epoch is inconsistent")
        if self.targets_per_update is not None and global_cursor % self.targets_per_update:
            raise ValueError("checkpoint global cursor is not update-aligned")
        self.global_cursor = global_cursor
        self.cursor = self.global_cursor % self.n_tokens
        self.epoch = self.global_cursor // self.n_tokens
