"""
Models package — re-exports all data models for backward-compatible imports.

Usage remains: `from src.models import ProductSpec, RouteDecision, ...`
"""

from src.models.enums import ChunkType, ContentType, SourceType
from src.models.ingestion import (
    DocMetadata,
    DocxContent,
    ExtractedSpec,
    ImageExtractionResult,
    Paragraph,
    ProductSpec,
)
from src.models.observability import RetrievalTrace, Source
from src.models.reports import IngestionReport
from src.models.retrieval import PreRouteSignals, RouteDecision
from src.models.vector import ChunkMetadata, DocumentChunk

__all__ = [
    "ChunkMetadata",
    "ChunkType",
    "ContentType",
    "DocMetadata",
    "DocxContent",
    "DocumentChunk",
    "ExtractedSpec",
    "ImageExtractionResult",
    "IngestionReport",
    "Paragraph",
    "PreRouteSignals",
    "ProductSpec",
    "RetrievalTrace",
    "RouteDecision",
    "Source",
    "SourceType",
]
