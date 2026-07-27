from __future__ import annotations

from pathlib import Path

import pytest

from scripts.platform_preflight import (
    LockValidationError,
    lock_for_runtime,
    validate_base_requirements,
    validate_lock_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements"


def test_base_requirements_and_platform_locks_are_exact_and_hashed():
    base = validate_base_requirements(REQUIREMENTS / "base.in")
    mac = validate_lock_file(
        REQUIREMENTS / "macos-arm64-py312.lock",
        expected_platform="macos-arm64",
    )
    linux = validate_lock_file(
        REQUIREMENTS / "linux-x86_64-cuda-py312.lock",
        expected_platform="linux-x86_64-cuda",
    )

    assert {"torch", "numpy", "tiktoken", "pyyaml", "pytest"} <= set(base)
    assert set(base) <= set(mac)
    assert set(base) <= set(linux)
    assert all(version and not version.startswith(("~", ">", "<", "!")) for version in mac.values())
    assert all(version and not version.startswith(("~", ">", "<", "!")) for version in linux.values())


@pytest.mark.parametrize(
    "content",
    [
        "# target-platform: macos-arm64\n# python-version: 3.12\nnumpy>=2\n",
        "# target-platform: macos-arm64\n# python-version: 3.12\nnumpy==2.0\n",
        (
            "# target-platform: macos-arm64\n# python-version: 3.12\n"
            "numpy==2.0 ; python_version >= '3.12' \\\n"
            "    --hash=sha256:" + "1" * 64 + "\n"
        ),
        (
            "# target-platform: macos-arm64\n# python-version: 3.12\n"
            "numpy==2.0 \\\n"
            "    --hash=sha256:" + "1" * 63 + "\n"
        ),
    ],
)
def test_lock_validation_rejects_ranges_unhashed_markers_and_bad_hashes(
    tmp_path,
    content,
):
    path = tmp_path / "bad.lock"
    path.write_text(content)

    with pytest.raises(LockValidationError):
        validate_lock_file(path, expected_platform="macos-arm64")


@pytest.mark.parametrize(
    "extra_header",
    [
        "# target-platform: macos-arm64\n",
        "# accelerator: cuda\n",
        "--index-url https://pypi.org/simple\n",
    ],
)
def test_lock_validation_rejects_ambiguous_metadata_and_index(
    tmp_path,
    extra_header,
):
    path = tmp_path / "macos-arm64-py312.lock"
    path.write_text(
        "# lock-format: uv-requirements-v1\n"
        "# target-platform: macos-arm64\n"
        "# python-version: 3.12\n"
        "# accelerator: mps\n"
        "--index-url https://pypi.org/simple\n"
        f"{extra_header}"
        "numpy==2.5.1 \\\n"
        f"    --hash=sha256:{'1' * 64}\n"
    )

    with pytest.raises(LockValidationError):
        validate_lock_file(path, expected_platform="macos-arm64")


def test_platform_lock_selection_is_unambiguous_and_python_312_only():
    assert lock_for_runtime(
        REQUIREMENTS,
        python_version=(3, 12, 9),
        system="Darwin",
        machine="arm64",
    ).name == "macos-arm64-py312.lock"
    assert lock_for_runtime(
        REQUIREMENTS,
        python_version=(3, 12, 0),
        system="Linux",
        machine="x86_64",
    ).name == "linux-x86_64-cuda-py312.lock"

    with pytest.raises(LockValidationError, match="Python 3.12"):
        lock_for_runtime(
            REQUIREMENTS,
            python_version=(3, 13, 0),
            system="Linux",
            machine="x86_64",
        )
    with pytest.raises(LockValidationError, match="unsupported|platform"):
        lock_for_runtime(
            REQUIREMENTS,
            python_version=(3, 12, 0),
            system="Darwin",
            machine="x86_64",
        )


def test_environment_setup_installs_one_explicit_lock_offline():
    setup = (REPO_ROOT / "cluster" / "setup_env.sh").read_text()

    assert "lock_for_runtime" in setup or "macos-arm64-py312.lock" in setup
    assert "linux-x86_64-cuda-py312.lock" in setup
    assert "--require-hashes" in setup
    assert "--offline" in setup
    assert "requirements.txt" not in setup
