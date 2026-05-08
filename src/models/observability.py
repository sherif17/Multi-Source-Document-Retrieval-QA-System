"""
Data models for observability: citations, traces, and answer explainability.
"""

from typing import Literal, Optional

from pydantic import BaseModel


class Source(BaseModel):
    """A citation source for answer explainability."""

    store: Literal["sql", "vector", "image_ocr"]
    content: str
    file: str
    detail: str
    relevance_score: Optional[float] = None


class RetrievalTrace(BaseModel):
    """One step in the observable retrieval execution trace."""

    node: str
    action: str
    duration_ms: float
    result_count: int = 0
    detail: Optional[str] = None
