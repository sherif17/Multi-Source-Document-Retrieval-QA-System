"""
Shared enumerations used across the application.
"""

from enum import Enum


class ContentType(str, Enum):
    """Classification of a DOCX paragraph's content type."""

    METADATA = "metadata"
    STRUCTURED = "structured"
    NARRATIVE = "narrative"
    IMAGE = "image"
    IGNORE = "ignore"


class SourceType(str, Enum):
    """Origin of a product specification record."""

    XLSX = "xlsx"
    DOCX_TEXT = "docx_text"
    DOCX_IMAGE = "docx_image"


class ChunkType(str, Enum):
    """Type of content stored in the vector database."""

    NARRATIVE = "narrative"
    SPEC_CONTEXT = "spec_context"
    IMAGE_OCR = "image_ocr"
