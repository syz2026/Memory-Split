"""Strict, shell-safe profile contract for paired Slurm allocations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PYTHON_TEMPLATE = "${MS135_VENV}/bin/python"
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "profile_id",
        "platform",
        "site_id",
        "partition",
        "account",
        "qos",
        "gres",
        "gpu_name_regex",
        "gpus_per_pair",
        "cpus",
        "memory_gb",
        "wall_minutes",
        "python",
        "environment_allowlist",
    }
)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_GRES_RE = re.compile(r"^gpu(?::[A-Za-z0-9][A-Za-z0-9_.-]{0,63})?:2$")
_GPU_REGEX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.|()+-]{0,127}$")
_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_FORBIDDEN = ("\x00", "\n", "\r", "\t", ";", "&", "<", ">", "`", "$(")


@dataclass(frozen=True)
class SlurmProfile:
    schema_version: int
    profile_id: str
    platform: str
    site_id: str
    partition: str
    account: str | None
    qos: str | None
    gres: str
    gpu_name_regex: str
    gpus_per_pair: int
    cpus: int
    memory_gb: int
    wall_minutes: int
    python: str
    environment_allowlist: tuple[str, ...]
    sha256: str
    source_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "platform": self.platform,
            "site_id": self.site_id,
            "partition": self.partition,
            "account": self.account,
            "qos": self.qos,
            "gres": self.gres,
            "gpu_name_regex": self.gpu_name_regex,
            "gpus_per_pair": self.gpus_per_pair,
            "cpus": self.cpus,
            "memory_gb": self.memory_gb,
            "wall_minutes": self.wall_minutes,
            "python": self.python,
            "environment_allowlist": list(self.environment_allowlist),
        }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"profile contains duplicate key: {key}")
        result[key] = value
    return result


def _safe_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if (
        any(marker in value for marker in _FORBIDDEN)
        or "/" in value
        or "\\" in value
        or ".." in value
        or value.startswith("~")
    ):
        raise ValueError(f"{field} contains a path or shell metacharacter")
    return value


def _name(value: object, *, field: str, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    text = _safe_text(value, field=field)
    if not _NAME_RE.fullmatch(text):
        raise ValueError(f"{field} contains unsupported characters")
    return text


def _positive_int(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > maximum
    ):
        raise ValueError(f"{field} must be an integer between 1 and {maximum}")
    return value


def validate_profile(
    raw: object,
    *,
    sha256: str,
    source_path: Path | str = Path("<memory>"),
) -> SlurmProfile:
    if not isinstance(raw, dict):
        raise ValueError("profile must contain a JSON object")  # noqa: TRY004
    missing = sorted(REQUIRED_FIELDS - set(raw))
    unknown = sorted(set(raw) - REQUIRED_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"profile fields do not match schema; missing={missing}, unknown={unknown}"
        )
    if raw["schema_version"] != SCHEMA_VERSION or isinstance(
        raw["schema_version"], bool
    ):
        raise ValueError("schema_version must be exactly 1")
    platform = raw["platform"]
    if platform not in {"aws", "farmshare", "mit"}:
        raise ValueError("platform must be aws, farmshare, or mit")
    gres = _safe_text(raw["gres"], field="gres")
    if not _GRES_RE.fullmatch(gres):
        raise ValueError("gres must request exactly two GPUs")
    gpu_regex = _safe_text(raw["gpu_name_regex"], field="gpu_name_regex")
    if not _GPU_REGEX_RE.fullmatch(gpu_regex):
        raise ValueError("gpu_name_regex contains unsupported characters")
    try:
        re.compile(gpu_regex)
    except re.error as error:
        raise ValueError("gpu_name_regex is invalid") from error
    if raw["gpus_per_pair"] != 2 or isinstance(raw["gpus_per_pair"], bool):
        raise ValueError("gpus_per_pair must be exactly 2")
    if raw["python"] != PYTHON_TEMPLATE:
        raise ValueError(f"python must be exactly {PYTHON_TEMPLATE}")
    allowlist = raw["environment_allowlist"]
    if (
        not isinstance(allowlist, list)
        or not allowlist
        or len(allowlist) != len(set(allowlist))
        or any(not isinstance(item, str) or not _ENV_RE.fullmatch(item) for item in allowlist)
    ):
        raise ValueError("environment_allowlist must contain unique safe names")
    required_environment = {
        "HOME",
        "PATH",
        "PYTHONPATH",
        "MS135_VENV",
        "SLURM_JOB_ID",
        "SLURM_SUBMIT_DIR",
    }
    if not required_environment <= set(allowlist):
        raise ValueError("environment_allowlist omits required runtime names")
    return SlurmProfile(
        schema_version=SCHEMA_VERSION,
        profile_id=str(_name(raw["profile_id"], field="profile_id")),
        platform=platform,
        site_id=str(_name(raw["site_id"], field="site_id")),
        partition=str(_name(raw["partition"], field="partition")),
        account=_name(raw["account"], field="account", optional=True),
        qos=_name(raw["qos"], field="qos", optional=True),
        gres=gres,
        gpu_name_regex=gpu_regex,
        gpus_per_pair=2,
        cpus=_positive_int(raw["cpus"], field="cpus", maximum=1024),
        memory_gb=_positive_int(
            raw["memory_gb"],
            field="memory_gb",
            maximum=1_048_576,
        ),
        wall_minutes=_positive_int(
            raw["wall_minutes"],
            field="wall_minutes",
            maximum=43_200,
        ),
        python=PYTHON_TEMPLATE,
        environment_allowlist=tuple(allowlist),
        sha256=sha256,
        source_path=Path(source_path),
    )


def load_profile(path: Path | str) -> SlurmProfile:
    profile_path = Path(path)
    if not profile_path.is_file() or profile_path.is_symlink():
        raise ValueError(f"profile is missing or unsafe: {profile_path}")
    data = profile_path.read_bytes()
    if len(data) > 65_536:
        raise ValueError("profile exceeds 64 KiB")
    try:
        raw = json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"profile contains non-finite value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("profile must contain valid UTF-8 JSON") from error
    return validate_profile(
        raw,
        sha256=hashlib.sha256(data).hexdigest(),
        source_path=profile_path.resolve(),
    )
