"""Deterministic metadata-first parallel corpus construction."""

from .adapters import UnsupportedProductionRenderer, UnsupportedSourceError
from .catalog import CatalogRecord, InputCatalog, fixture_catalog
from .integrity import ordered_stream_commitments, verify_stream_commitments
from .metadata import (
    FixtureRenderer,
    MetadataRecord,
    RenderedRecord,
    Renderer,
    metadata_from_bytes,
    metadata_to_bytes,
    reduce_metadata,
    render_metadata,
)
from .publication import (
    ParallelBuildConfig,
    build_parallel_corpus,
    parallel_build_id,
    publication_staging_path,
    rerender_and_pack,
    verify_parallel_corpus,
)
from .schedule import (
    ScheduleRecord,
    ShardAssignment,
    assignments_from_bytes,
    assignments_to_bytes,
    assign_update_aligned_shards,
    largest_deficit_schedule,
    schedule_from_bytes,
    schedule_to_bytes,
)

__all__ = [
    "CatalogRecord",
    "FixtureRenderer",
    "InputCatalog",
    "MetadataRecord",
    "ParallelBuildConfig",
    "RenderedRecord",
    "Renderer",
    "ScheduleRecord",
    "ShardAssignment",
    "UnsupportedProductionRenderer",
    "UnsupportedSourceError",
    "assign_update_aligned_shards",
    "assignments_from_bytes",
    "assignments_to_bytes",
    "build_parallel_corpus",
    "fixture_catalog",
    "largest_deficit_schedule",
    "metadata_from_bytes",
    "metadata_to_bytes",
    "ordered_stream_commitments",
    "parallel_build_id",
    "publication_staging_path",
    "reduce_metadata",
    "render_metadata",
    "rerender_and_pack",
    "schedule_from_bytes",
    "schedule_to_bytes",
    "verify_parallel_corpus",
    "verify_stream_commitments",
]
