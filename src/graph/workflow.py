"""
LangGraph workflow construction — wires all nodes into an executable graph.

This is the central orchestration engine. It defines:
- Node registration
- Edge connections (linear and conditional)
- Routing logic (which retriever to invoke based on route decision)
- Retry loop (guardrails failure → widen retrieval → re-synthesize)

The compiled graph is the single entry point called by the Streamlit UI.
Invoke with an initial state dict, get back the final state with answer + trace.
"""

from langgraph.graph import END, StateGraph

from src.utils.logger import logger

from .nodes import (
    answer_synthesizer,
    guardrails_checker,
    hybrid_retriever,
    llm_router,
    pre_router,
    query_analyzer,
    structured_retriever,
    unstructured_retriever,
)
from .state import GraphState


# ═══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def route_to_retriever(state: GraphState) -> str:
    """
    After llm_router: select which retriever node to invoke.

    Logic:
    - Low confidence (< 0.5) → hybrid (safe fallback)
    - structured_only → structured_retriever
    - unstructured_only → unstructured_retriever
    - Everything else → hybrid_retriever
    """
    confidence = state.get("route_confidence", 0.5)
    strategy = state.get("route", "hybrid_sql_primary")

    if confidence < 0.5:
        logger.info("Low confidence routing → defaulting to hybrid retriever")
        return "hybrid_retriever"

    if strategy == "structured_only":
        return "structured_retriever"
    elif strategy == "unstructured_only":
        return "unstructured_retriever"
    else:
        return "hybrid_retriever"


def check_guardrails(state: GraphState) -> str:
    """
    After guardrails_checker: decide whether to retry or finish.

    If guardrails detected issues and retry budget remains,
    route back to hybrid_retriever for wider retrieval + re-synthesis.
    """
    if state.get("needs_fallback") and state.get("retry_count", 0) <= 2:
        logger.info("Guardrails triggered retry → widening retrieval")
        return "hybrid_retriever"
    return END


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════


def build_graph() -> StateGraph:
    """
    Construct the full LangGraph workflow.

    Flow:
        query_analyzer → pre_router → llm_router
            → [structured | unstructured | hybrid]_retriever
            → answer_synthesizer → guardrails_checker
            → END (or retry → hybrid_retriever → answer_synthesizer → ...)
    """
    graph = StateGraph(GraphState)

    # Register all nodes
    graph.add_node("query_analyzer", query_analyzer)
    graph.add_node("pre_router", pre_router)
    graph.add_node("llm_router", llm_router)
    graph.add_node("structured_retriever", structured_retriever)
    graph.add_node("unstructured_retriever", unstructured_retriever)
    graph.add_node("hybrid_retriever", hybrid_retriever)
    graph.add_node("answer_synthesizer", answer_synthesizer)
    graph.add_node("guardrails_checker", guardrails_checker)

    # Linear edges (always execute in sequence)
    graph.set_entry_point("query_analyzer")
    graph.add_edge("query_analyzer", "pre_router")
    graph.add_edge("pre_router", "llm_router")

    # Conditional: llm_router → retriever selection
    graph.add_conditional_edges(
        "llm_router",
        route_to_retriever,
        {
            "structured_retriever": "structured_retriever",
            "unstructured_retriever": "unstructured_retriever",
            "hybrid_retriever": "hybrid_retriever",
        },
    )

    # All retrievers converge to synthesis
    graph.add_edge("structured_retriever", "answer_synthesizer")
    graph.add_edge("unstructured_retriever", "answer_synthesizer")
    graph.add_edge("hybrid_retriever", "answer_synthesizer")

    # Synthesis → guardrails
    graph.add_edge("answer_synthesizer", "guardrails_checker")

    # Conditional: guardrails → END or retry
    graph.add_conditional_edges(
        "guardrails_checker",
        check_guardrails,
        {
            "hybrid_retriever": "hybrid_retriever",
            END: END,
        },
    )

    return graph


# ═══════════════════════════════════════════════════════════════════════════════
# COMPILED WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════


def get_workflow():
    """
    Get the compiled, ready-to-invoke workflow (full pipeline including synthesis).

    Usage:
        workflow = get_workflow()
        result = workflow.invoke({
            "query": "What is the max VOC for Aurora interior paint in EU?",
            "chat_history": [],
            "session_id": "abc123",
            "retrieval_trace": [],
            "retry_count": 0,
            "needs_fallback": False,
        })
        print(result["answer"])
    """
    graph = build_graph()
    workflow = graph.compile()
    logger.info("LangGraph workflow compiled successfully")
    return workflow


def build_retrieval_graph() -> StateGraph:
    """
    Build a retrieval-only workflow (no synthesis or guardrails).

    Used for the streaming UI path where synthesis is handled separately
    with OpenAI streaming tokens directly to Streamlit.
    """
    graph = StateGraph(GraphState)

    graph.add_node("query_analyzer", query_analyzer)
    graph.add_node("pre_router", pre_router)
    graph.add_node("llm_router", llm_router)
    graph.add_node("structured_retriever", structured_retriever)
    graph.add_node("unstructured_retriever", unstructured_retriever)
    graph.add_node("hybrid_retriever", hybrid_retriever)

    graph.set_entry_point("query_analyzer")
    graph.add_edge("query_analyzer", "pre_router")
    graph.add_edge("pre_router", "llm_router")

    graph.add_conditional_edges(
        "llm_router",
        route_to_retriever,
        {
            "structured_retriever": "structured_retriever",
            "unstructured_retriever": "unstructured_retriever",
            "hybrid_retriever": "hybrid_retriever",
        },
    )

    # All retrievers → END (no synthesis)
    graph.add_edge("structured_retriever", END)
    graph.add_edge("unstructured_retriever", END)
    graph.add_edge("hybrid_retriever", END)

    return graph


def get_retrieval_workflow():
    """Get the retrieval-only workflow (for streaming UI)."""
    graph = build_retrieval_graph()
    workflow = graph.compile()
    logger.info("Retrieval-only workflow compiled successfully")
    return workflow
