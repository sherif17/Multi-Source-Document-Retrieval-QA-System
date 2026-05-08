"""
Data models for pipeline reports and summaries.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class IngestionReport(BaseModel):
    """Summary of a complete ingestion pipeline run."""

    total_xlsx_rows: int = 0
    total_docx_specs_extracted: int = 0
    total_image_specs_extracted: int = 0
    total_narrative_chunks: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
