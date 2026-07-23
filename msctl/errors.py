"""Structured errors safe to return from the JSON CLI."""

from __future__ import annotations

from collections.abc import Mapping


class MsctlError(Exception):
    """Expected operational failure with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
        exit_code: int = 2,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})
        self.exit_code = exit_code

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }
