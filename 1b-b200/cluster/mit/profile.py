"""Compatibility wrapper for the generic paired Slurm profile."""

from msctl.profile import (  # noqa: F401
    PYTHON_TEMPLATE,
    SCHEMA_VERSION,
    SlurmProfile,
    load_profile,
    validate_profile,
)

MITProfile = SlurmProfile

__all__ = [
    "MITProfile",
    "PYTHON_TEMPLATE",
    "SCHEMA_VERSION",
    "SlurmProfile",
    "load_profile",
    "validate_profile",
]
