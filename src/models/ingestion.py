"""
Data models for the ingestion pipeline.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.models.enums import SourceType


class ProductSpec(BaseModel):
    """
    Canonical product specification record — unified schema for all clients.

    This is the single data model written to Neon DB. Both Aurora and Horizon
    XLSX schemas are normalized into this shape during ingestion.
    """

    client_id: str
    product_name: str
    region: str
    parameter: str
    value: Optional[float] = None
    unit: Optional[str] = None
    limit_type: Optional[str] = None
    notes: Optional[str] = None
    source_file: str
    source_type: SourceType

    @field_validator("client_id", mode="before")
    @classmethod
    def normalize_client_id(cls, v: str) -> str:
        """Normalize client identifiers to lowercase canonical form."""
        mapping = {
            "aurora paints": "aurora",
            "aurora": "aurora",
            "horizon coatings": "horizon",
            "horizon": "horizon",
        }
        normalized = mapping.get(v.lower().strip())
        if normalized:
            return normalized
        # Fuzzy fallback
        lower = v.lower()
        if "aurora" in lower:
            return "aurora"
        if "horizon" in lower:
            return "horizon"
        raise ValueError(f"Unknown client: {v}")

    @field_validator("region", mode="before")
    @classmethod
    def normalize_region(cls, v: str) -> str:
        """Normalize region identifiers."""
        mapping = {
            "eu": "EU",
            "european union": "EU",
            "europe": "EU",
            "us": "US",
            "usa": "US",
            "united states": "US",
            "gcc": "GCC",
            "gulf": "GCC",
        }
        return mapping.get(v.lower().strip(), v.upper().strip())

    @field_validator("unit", mode="before")
    @classmethod
    def normalize_unit(cls, v: Optional[str]) -> Optional[str]:
        """Normalize measurement units."""
        if v is None:
            return None
        mapping = {
            "g/l": "g/L",
            "g/L": "g/L",
            "% by weight": "% by weight",
            "%": "% by weight",
            "hours": "hours",
            "hour": "hours",
        }
        return mapping.get(v.strip(), v.strip())

    @field_validator("limit_type", mode="before")
    @classmethod
    def normalize_limit_type(cls, v: Optional[str]) -> Optional[str]:
        """Normalize limit type classifications."""
        if v is None:
            return None
        mapping = {
            "internal_limit": "internal_limit",
            "client_internal_limit": "internal_limit",
            "illustrative_internal_limit": "illustrative_internal_limit",
            "application_guideline": "application_guideline",
        }
        return mapping.get(v.strip(), v.strip())


class DocMetadata(BaseModel):
    """Metadata extracted from the header paragraphs of a DOCX file."""

    client: str
    product_family: Optional[str] = None
    product_variant: Optional[str] = None


class Paragraph(BaseModel):
    """A single paragraph extracted from a DOCX file."""

    index: int
    text: str
    style: str
    has_image: bool = False


class DocxContent(BaseModel):
    """Complete parsed content of a DOCX file."""

    file_path: str
    paragraphs: list[Paragraph]
    metadata: DocMetadata
    image_count: int = 0


class ExtractedSpec(BaseModel):
    """A specification value extracted from DOCX text or image via regex/LLM."""

    product_name: Optional[str] = None
    region: Optional[str] = None
    parameter: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "medium"


class ImageExtractionResult(BaseModel):
    """Result from the dual OCR/Vision pipeline on an embedded image."""

    tesseract_text: str = ""
    vision_specs: list[ExtractedSpec] = Field(default_factory=list)
    final_specs: list[ExtractedSpec] = Field(default_factory=list)
    agreement_score: float = 0.0
