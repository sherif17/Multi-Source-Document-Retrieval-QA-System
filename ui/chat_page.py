"""
Chat page — query processing, streaming responses, and info panels.
"""

import streamlit as st

from src.utils.logger import logger


# ═══════════════════════════════════════════════════════════════════════════════
# NODE LABELS (for progress display)
# ═══════════════════════════════════════════════════════════════════════════════

_NODE_LABELS = {
    "query_analyzer": "Analyzing query...",
    "pre_router": "Extracting signals...",
    "llm_router": "Routing to retriever...",
    "structured_retriever": "Querying database...",
    "unstructured_retriever": "Searching documents...",
    "hybrid_retriever": "Hybrid retrieval (SQL + vector)...",
    "conversational_responder": "Responding...",
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════


def render_chat_page(workflow):
    """Render the Chat page with query processing and info panels."""
    st.title("Multi-Source Document Retrieval & QA")

    # Check if data has been ingested
    if not st.session_state.ingestion_done:
        st.warning(
            "No data has been ingested yet. "
            "Please go to **Upload & Ingest** first to upload and process your documents."
        )
        st.stop()

    # Show auto-detection notice (once)
    if "_auto_detected" in st.session_state:
        state = st.session_state.pop("_auto_detected")
        st.info(
            f"Connected to existing stores — "
            f"{state['total_rows']} DB rows, {state['total_vectors']} vectors. "
            f"Ready to query."
        )

    _render_example_queries()
    _render_clear_button()
    _render_chat_history()

    user_input = _get_user_input()

    if user_input:
        _process_query(user_input, workflow)

    _render_info_panels()


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════


def _render_example_queries():
    """Show expandable example queries."""
    with st.expander("Example Queries", expanded=False):
        examples = [
            "For client Aurora Paints, what is the maximum lead content allowed in EcoSafe Interior Wall Paint for the EU?",
            "For client Aurora Paints, considering EcoSafe Ceiling Paint, EcoSafe Exterior Facade and EcoShield Floor Coating in the EU, what is the maximum internal VOC limit in g/L across these products?",
            "According to the guidance for client Horizon Coatings' UltraSafe Interior Wall Paint, in which types of rooms is enhanced ventilation recommended?",
            "Comparing Aurora Paints and Horizon Coatings, which client sets a stricter VOC limit for interior wall paint in the EU?",
            "For client Aurora Paints, what internal VOC limit in g/L is set for EcoSafe Kitchen & Bath in the EU for typical residential projects?",
        ]
        for i, example in enumerate(examples):
            if st.button(f"Query {i+1}", key=f"example_{i}", help=example):
                st.session_state.example_query = example
                st.rerun()


def _render_clear_button():
    """Render the Clear conversation button."""
    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.last_trace = None
            st.session_state.last_sources = None
            st.session_state.last_route_info = None
            st.rerun()


def _render_chat_history():
    """Display existing chat messages."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _get_user_input() -> str | None:
    """Get user input from example query or chat input."""
    if "example_query" in st.session_state:
        return st.session_state.pop("example_query")
    return st.chat_input("Ask a question about product specifications...")


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════


def _process_query(user_input: str, workflow):
    """Run the full query pipeline: route → retrieve → stream response."""
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()

        try:
            from langchain_core.messages import AIMessage, HumanMessage
            from src.graph.nodes import stream_synthesis, stream_conversational

            # Phase 1: Routing & Retrieval with live progress
            retrieval_result, is_conversational = _run_retrieval(
                user_input, workflow, status_placeholder
            )

            # Clear status before streaming answer
            status_placeholder.empty()

            # Phase 2: Stream response
            if is_conversational:
                answer = st.write_stream(stream_conversational(retrieval_result))
            else:
                answer = st.write_stream(stream_synthesis(retrieval_result))

            # Store metadata panels (only meaningful for data queries)
            if not is_conversational:
                st.session_state.last_trace = retrieval_result.get("retrieval_trace", [])
                st.session_state.last_sources = retrieval_result.get("sources", [])
                st.session_state.last_route_info = {
                    "route": retrieval_result.get("route", "unknown"),
                    "reasoning": retrieval_result.get("route_reasoning", ""),
                    "confidence": retrieval_result.get("route_confidence", 0),
                    "intent": retrieval_result.get("route_decision", {}).get("intent", "unknown"),
                    "clients": retrieval_result.get("extracted_clients", []),
                }

            # Update conversation history (both paths)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=answer))

            # Trim history window
            from src.config import settings
            max_window = settings.chat_history_window
            if len(st.session_state.chat_history) > max_window:
                st.session_state.chat_history = st.session_state.chat_history[-max_window:]

        except Exception as e:
            status_placeholder.empty()
            error_msg = f"An error occurred: {str(e)}"
            st.error(error_msg)
            logger.error(f"Workflow error: {e}", exc_info=True)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})


def _run_retrieval(user_input: str, workflow, status_placeholder) -> tuple[dict, bool]:
    """Execute the LangGraph workflow and stream progress updates."""
    retrieval_result = {}
    is_conversational = False

    for event in workflow.stream(
        {
            "query": user_input,
            "chat_history": st.session_state.chat_history,
            "session_id": "streamlit_session",
            "retrieval_trace": [],
            "retry_count": 0,
            "needs_fallback": False,
        },
        stream_mode="updates",
    ):
        for node_name, state_update in event.items():
            label = _NODE_LABELS.get(node_name, f"{node_name}...")
            detail = _get_node_detail(node_name, state_update)

            if node_name == "conversational_responder":
                is_conversational = True

            status_placeholder.markdown(f"*{label}{detail}*")
            retrieval_result.update(state_update)

    return retrieval_result, is_conversational


def _get_node_detail(node_name: str, state_update: dict) -> str:
    """Extract display detail for a node's progress update."""
    if node_name == "llm_router" and state_update.get("route"):
        return f" > `{state_update['route']}`"
    elif node_name in ("structured_retriever", "hybrid_retriever"):
        rows = len(state_update.get("sql_results", []))
        vecs = len(state_update.get("vector_results", []))
        if rows or vecs:
            return f" > {rows} rows, {vecs} docs"
    elif node_name == "unstructured_retriever":
        vecs = len(state_update.get("vector_results", []))
        return f" > {vecs} results"
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# INFORMATION PANELS
# ═══════════════════════════════════════════════════════════════════════════════


def _render_info_panels():
    """Render routing decision, sources, and retrieval trace panels."""
    if not (st.session_state.last_route_info or st.session_state.last_trace):
        return

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.last_route_info:
            with st.expander("Routing Decision", expanded=False):
                info = st.session_state.last_route_info
                st.markdown(f"**Intent:** {info.get('intent', '?')}")
                st.markdown(f"**Strategy:** `{info.get('route', '?')}`")
                st.markdown(f"**Confidence:** {info.get('confidence', 0):.0%}")
                st.markdown(f"**Client(s):** {', '.join(info.get('clients', []))}")
                st.markdown(f"**Reasoning:** {info.get('reasoning', 'N/A')}")

    with col2:
        if st.session_state.last_sources:
            with st.expander("Sources", expanded=False):
                for i, source in enumerate(st.session_state.last_sources, 1):
                    store = source.get("store", "?")
                    tag = {"sql": "[SQL]", "vector": "[VEC]", "image_ocr": "[IMG]"}.get(store, "[?]")
                    file = source.get("file", "unknown")
                    detail = source.get("detail", "")
                    score = source.get("relevance_score")
                    score_text = f" | score: {score:.2f}" if score else ""
                    st.markdown(f"{tag} **[{i}]** `{file}` — {detail}{score_text}")

    if st.session_state.last_trace:
        with st.expander("Retrieval Trace", expanded=False):
            from src.utils.citations import format_trace_for_display
            trace = format_trace_for_display(st.session_state.last_trace)
            total_ms = sum(step["duration_ms"] for step in trace)
            st.markdown(f"**Total execution time:** {total_ms:.0f}ms")
            st.markdown("---")

            for i, step in enumerate(trace, 1):
                node = step["node"]
                action = step["action"]
                duration = step["duration_ms"]
                detail = step.get("detail")

                st.markdown(f"**{i}. {node}** ({duration:.0f}ms)")
                st.markdown(f"   → {action}")
                if detail:
                    if "SELECT" in str(detail).upper():
                        st.code(detail, language="sql")
                    else:
                        st.caption(detail)
