from __future__ import annotations

import os
import pickle
import time

import pytest

from cluster.aws.p5.checkpoint_mirror import (
    CheckpointReceiptRef,
    ForkedCheckpointMirrorAttempt,
    PublishedCheckpointPair,
    VersionedUploadedObject,
)


def _published_pair() -> PublishedCheckpointPair:
    return PublishedCheckpointPair(
        receipt=CheckpointReceiptRef(
            uri="s3://bucket/receipt.json",
            sha256="a" * 64,
            version_id="receipt-version",
            bytes=123,
        ),
        checkpoints=(
            VersionedUploadedObject(
                "s3://bucket/dense.pt",
                "b" * 64,
                10,
                "dense-version",
            ),
            VersionedUploadedObject(
                "s3://bucket/split90.pt",
                "c" * 64,
                10,
                "split90-version",
            ),
        ),
        value={"fixture": "pair"},
    )


def _poll_until_complete(
    attempt: ForkedCheckpointMirrorAttempt,
) -> tuple[bool, PublishedCheckpointPair | None]:
    deadline = time.monotonic() + 30.0
    complete, result = attempt.poll()
    while not complete and time.monotonic() < deadline:
        time.sleep(0.02)
        complete, result = attempt.poll()
    return complete, result


def _raw_forked_attempt(
    payload: bytes,
    *,
    exit_code: int,
    cancel_cleanup,
) -> ForkedCheckpointMirrorAttempt:
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        view = memoryview(payload)
        while view:
            written = os.write(write_fd, view)
            view = view[written:]
        os.close(write_fd)
        os._exit(exit_code)
    os.close(write_fd)
    os.set_blocking(read_fd, False)
    attempt = ForkedCheckpointMirrorAttempt.__new__(
        ForkedCheckpointMirrorAttempt
    )
    attempt._pid = pid
    attempt._read_fd = read_fd
    attempt._chunks = []
    attempt._saw_eof = False
    attempt._cancel_cleanup = cancel_cleanup
    return attempt


@pytest.mark.parametrize(
    "failure",
    [
        "child-reported",
        "wrong-type",
        "malformed",
        "truncated",
        "abnormal-exit",
    ],
)
def test_forked_attempt_failure_runs_cleanup_exactly_once(failure) -> None:
    cleanups = []

    def cleanup() -> None:
        cleanups.append("cleanup")

    expected = _published_pair()
    if failure == "child-reported":
        attempt = ForkedCheckpointMirrorAttempt(
            lambda: (_ for _ in ()).throw(RuntimeError("publish failed")),
            cancel_cleanup=cleanup,
        )
    elif failure == "wrong-type":
        attempt = ForkedCheckpointMirrorAttempt(
            lambda: "not-a-published-pair",
            cancel_cleanup=cleanup,
        )
    elif failure == "malformed":
        attempt = _raw_forked_attempt(
            b"not-a-pickle",
            exit_code=0,
            cancel_cleanup=cleanup,
        )
    elif failure == "truncated":
        payload = pickle.dumps((True, expected), protocol=5)
        attempt = _raw_forked_attempt(
            payload[: len(payload) // 2],
            exit_code=0,
            cancel_cleanup=cleanup,
        )
    elif failure == "abnormal-exit":
        attempt = _raw_forked_attempt(
            pickle.dumps((True, expected), protocol=5),
            exit_code=7,
            cancel_cleanup=cleanup,
        )
    else:
        raise AssertionError(failure)

    complete, result = _poll_until_complete(attempt)

    assert complete is True
    assert result is None
    assert cleanups == ["cleanup"]
    attempt.cancel()
    attempt.cancel()
    assert cleanups == ["cleanup"]


def test_forked_attempt_valid_success_clears_cleanup_without_invoking_it() -> None:
    cleanups = []
    expected = _published_pair()
    attempt = ForkedCheckpointMirrorAttempt(
        lambda: expected,
        cancel_cleanup=lambda: cleanups.append("cleanup"),
    )

    complete, result = _poll_until_complete(attempt)

    assert complete is True
    assert result == expected
    attempt.cancel()
    attempt.cancel()
    assert cleanups == []
