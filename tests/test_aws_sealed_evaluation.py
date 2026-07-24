from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from msctl.aws_sealed_evaluation import (
    REQUIRED_SEALED_MEMBERS,
    load_sealed_evaluation_release,
    main,
)
from msctl.errors import MsctlError
from tests.test_v3_hardware_amendment import _sealed_release


def test_external_sealed_release_binds_actual_study_lock_and_exact_inventory(
    tmp_path,
    capsys,
):
    release_root = _sealed_release(tmp_path)
    release = load_sealed_evaluation_release(release_root)

    assert set(release.members) == set(REQUIRED_SEALED_MEMBERS)
    assert release.study_lock_sha256 == hashlib.sha256(
        (release_root / "study-lock.json").read_bytes()
    ).hexdigest()
    assert release.sha256 != release.study_lock_sha256

    assert (
        main(
            [
                "--root",
                str(release_root),
                "--expected-release-sha256",
                release.sha256,
                "--expected-study-lock-sha256",
                release.study_lock_sha256,
            ]
        )
        == 0
    )
    assert release.study_lock_sha256 in capsys.readouterr().out


@pytest.mark.parametrize("change", ["missing", "extra", "wrong-lock"])
def test_external_sealed_release_rejects_open_or_stale_contract(tmp_path, change):
    release_root = _sealed_release(tmp_path)
    expected = load_sealed_evaluation_release(release_root)

    if change == "missing":
        (release_root / "checkpoints.jsonl").unlink()
    elif change == "extra":
        (release_root / "unreviewed.json").write_text("{}\n", encoding="ascii")

    with pytest.raises(MsctlError):
        load_sealed_evaluation_release(
            release_root,
            expected_release_sha256=expected.sha256,
            expected_study_lock_sha256=(
                "f" * 64
                if change == "wrong-lock"
                else expected.study_lock_sha256
            ),
        )
