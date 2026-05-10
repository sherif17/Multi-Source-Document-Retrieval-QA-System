"""
Data models for query routing and retrieval decisions.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Output of the LLM query router — determines retrieval strategy."""

    intent: Literal["lookup", "aggregation", "narrative", "comparison", "hybrid", "conversational"]
    strategy: Literal[
        "structured_only",
        "unstructured_only",
        "hybrid_sql_primary",
        "hybrid_vec_primary",
        "cross_client_hybrid",
        "conversational",
    ]
    clients: list[str]
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    sub_queries: list[str] = Field(default_factory=list)
    needs_aggregation: bool = False
    needs_narrative_context: bool = False


class PreRouteSignals(BaseModel):
    """Deterministic signals extracted by the rule-based pre-router."""

    extracted_clients: list[str]
    has_numeric_ask: bool = False
    has_aggregation: bool = False
    has_narrative_ask: bool = False
    is_cross_client: bool = False
    has_product_list: bool = False
