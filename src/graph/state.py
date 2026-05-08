"""
LangGraph state definition — the central data contract for the orchestration workflow.

Every node in the graph reads from and writes to this state object.
TypedDict is used (not dataclass) because LangGraph requires it for
its state management and checkpointing system.

Design principle: State is the ONLY way nodes communicate.
No global variables, no side channels. This makes the graph:
- Fully observable (inspect state at any point)
- Reproducible (replay with same initial state)
- Debuggable (each node's input/output is visible)
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Complete state schema for the LangGraph orchestration workflow.

    Sections:
    - Input: Raw user query and conversation context
    - Pre-routing: Deterministic signal extraction results
    - Routing: LLM-based route decision
    - Retrieval: Results from structured/unstructured stores
    - Synthesis: Generated answer and citations
    - Observability: Execution trace for UI display
    - Control: Retry logic and error state
    """

    # ── Input ──
    query: str
    chat_history: Annotated[list, add_messages]
    session_id: str

    # ── Pre-routing ──
    reformulated_query: str
    extracted_clients: list[str]
    pre_route_signals: dict[str, Any]

    # ── Routing ──
    route_decision: dict[str, Any]
    route: str
    route_reasoning: str
    route_confidence: float
    sub_queries: list[str]

    # ── Retrieval Results ──
    sql_query: Optional[str]
    sql_results: list[dict]
    sql_error: Optional[str]
    vector_results: list[dict]
    formatted_sql: str
    formatted_vector: str

    # ── Synthesis ──
    answer: str
    sources: list[dict]

    # ── Observability ──
    retrieval_trace: list[dict]

    # ── Control ──
    error: Optional[str]
    retry_count: int
    needs_fallback: bool
