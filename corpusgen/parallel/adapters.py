"""Fail-closed adapter boundary for production source renderers."""

from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import CatalogRecord
from .metadata import RenderedRecord


class UnsupportedSourceError(RuntimeError):
    """Raised when a production renderer has not been explicitly implemented."""


@dataclass(frozen=True)
class UnsupportedProductionRenderer:
    source_name: str
    renderer_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_name, str) or not self.source_name:
            raise ValueError("source_name must be a non-empty string")
        object.__setattr__(
            self,
            "renderer_id",
            f"unsupported-production:{self.source_name}:v1",
        )

    def render(self, record: CatalogRecord) -> RenderedRecord:
        raise UnsupportedSourceError(
            f"production renderer for {self.source_name!r} is unsupported; "
            f"refusing to render {record.record_id!r}"
        )
