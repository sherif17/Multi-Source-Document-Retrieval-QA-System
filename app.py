"""
Streamlit application for the Multi-Source Document Retrieval system.

Two-page layout:
1. Upload & Ingest — Upload XLSX/DOCX files and watch real-time ingestion
2. Chat — Query the ingested data with full observability

Architecture: Thin UI layer — all intelligence lives in src/graph/workflow.py.
This file only handles rendering and session state management.
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import logger

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Multi-Source Document Retrieval",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP: Auto-detect DB state & pre-compile workflow
# ═══════════════════════════════════════════════════════════════════════════════


@st.cache_resource(show_spinner=False)
def _init_workflow():
    """Pre-compile LangGraph retrieval workflow once at startup."""
    from src.graph.workflow import get_retrieval_workflow
    return get_retrieval_workflow()


@st.cache_resource(show_spinner=False)
def _check_stores_state() -> dict:
    """Check if BOTH Neon DB and Pinecone already have data."""
    result = {
        "has_data": False,
        "total_rows": 0,
        "total_vectors": 0,
        "db_connected": False,
        "pinecone_connected": False,
    }

    # Check Neon DB
    try:
        from src.stores.sql_store import get_table_stats
        stats = get_table_stats()
        if stats:
            result["total_rows"] = sum(s.get("row_count", 0) for s in stats.values())
            result["db_connected"] = True
    except Exception as e:
        logger.warning(f"Could not connect to Neon DB on startup: {e}")

    # Check Pinecone
    try:
        from src.stores.vector_store import get_index_stats
        vec_stats = get_index_stats()
        result["total_vectors"] = vec_stats.get("total_vectors", 0)
        result["pinecone_connected"] = True
    except Exception as e:
        logger.warning(f"Could not connect to Pinecone on startup: {e}")

    # Only mark as having data if both stores have content
    if result["total_rows"] > 0 and result["total_vectors"] > 0:
        result["has_data"] = True

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_trace" not in st.session_state:
    st.session_state.last_trace = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = None

if "last_route_info" not in st.session_state:
    st.session_state.last_route_info = None

if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = False

if "ingestion_log" not in st.session_state:
    st.session_state.ingestion_log = []

# Auto-detect existing data on startup
if not st.session_state.ingestion_done:
    stores_state = _check_stores_state()
    if stores_state["has_data"]:
        st.session_state.ingestion_done = True
        st.session_state["_auto_detected"] = stores_state
        logger.info(
            f"Auto-detected {stores_state['total_rows']} DB rows + "
            f"{stores_state['total_vectors']} vectors — skipping upload"
        )

# Pre-compile workflow (cached, runs once)
workflow = _init_workflow()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("Doc Retrieval QA")
    st.markdown("---")

    # Page navigation
    page = st.radio(
        "Navigate",
        ["Upload & Ingest", "Chat"],
        index=0 if not st.session_state.ingestion_done else 1,
    )

    st.markdown("---")
    st.markdown(
        "**Architecture:**\n"
        "- LangGraph orchestration\n"
        "- Neon DB (structured)\n"
        "- Pinecone (unstructured)\n"
        "- Tiered routing (rules + LLM)\n"
        "- 3-layer client isolation\n"
        "- Dual OCR (Vision + Tesseract)"
    )

    st.markdown("---")
    status_label = "Ingested" if st.session_state.ingestion_done else "Not ingested"
    st.markdown(f"**Data Status:** {status_label}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: UPLOAD & INGEST
# ═══════════════════════════════════════════════════════════════════════════════

if page == "Upload & Ingest":
    st.title("Upload & Ingest Documents")
    st.markdown(
        "Upload your client XLSX and DOCX files below. "
        "The system will automatically detect schemas, classify content, "
        "extract specifications, and populate both the SQL and vector stores."
    )

    st.markdown("---")

    # File upload area
    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=["xlsx", "docx"],
        accept_multiple_files=True,
        help="Upload XLSX (product specs) and DOCX (product briefs) files. "
        "Filenames should contain 'aurora' or 'horizon' for client detection.",
    )

    if uploaded_files:
        # Show uploaded file summary
        st.markdown("### Uploaded Files")
        cols = st.columns(len(uploaded_files))
        for i, f in enumerate(uploaded_files):
            with cols[i]:
                ftype = "[XLSX]" if f.name.endswith(".xlsx") else "[DOCX]"
                st.metric(label=f"{ftype} {f.name}", value=f"{f.size / 1024:.1f} KB")

        st.markdown("---")

        # Ingest button
        if st.button("Run Ingestion Pipeline", type="primary"):
            # Save uploaded files to temp directory
            tmp_dir = Path(tempfile.mkdtemp())
            file_paths: list[Path] = []

            for uploaded_file in uploaded_files:
                file_path = tmp_dir / uploaded_file.name
                file_path.write_bytes(uploaded_file.getvalue())
                file_paths.append(file_path)

            # Import the streaming pipeline
            from src.ingestion.pipeline import run_ingestion_streaming

            # Clear previous log
            st.session_state.ingestion_log = []

            # Progress UI
            st.markdown("### Ingestion Progress")
            progress_bar = st.progress(0.0)
            status_container = st.container()

            # Run ingestion with real-time progress
            with status_container:
                for event in run_ingestion_streaming(file_paths):
                    # Update progress bar
                    if event.progress > 0:
                        progress_bar.progress(event.progress)

                    # Render event based on status
                    if event.status == "start":
                        st.markdown(f"**{event.message}**")
                    elif event.status == "success":
                        st.success(event.message)
                        if event.detail:
                            st.caption(f"   ↳ {event.detail}")
                    elif event.status == "warning":
                        st.warning(event.message)
                    elif event.status == "error":
                        st.error(event.message)
                    elif event.status == "progress":
                        st.info(event.message)

                    # Log for later display
                    st.session_state.ingestion_log.append({
                        "phase": event.phase,
                        "status": event.status,
                        "message": event.message,
                        "detail": event.detail,
                    })

            # Mark ingestion as complete
            progress_bar.progress(1.0)
            st.session_state.ingestion_done = True
            st.balloons()

            st.markdown("---")
            st.success("Data is ready! Switch to the **Chat** page to start querying.")

    elif st.session_state.ingestion_done:
        # Show previous ingestion summary
        st.success("Data already ingested. Switch to **Chat** to query.")

        if st.session_state.ingestion_log:
            with st.expander("Previous Ingestion Log", expanded=False):
                for entry in st.session_state.ingestion_log:
                    prefix = {
                        "success": "[OK]",
                        "error": "[ERR]",
                        "warning": "[WARN]",
                        "start": "[...]",
                        "progress": "[...]",
                    }.get(entry["status"], "[-]")
                    st.markdown(f"{prefix} {entry['message']}")

        if st.button("Re-ingest (upload new files)"):
            # Clear both stores
            with st.spinner("Clearing existing data from both stores..."):
                try:
                    from src.stores.sql_store import truncate_specs
                    truncate_specs()
                except Exception as e:
                    logger.warning(f"Could not truncate DB: {e}")
                try:
                    from src.stores.vector_store import delete_namespace
                    delete_namespace("aurora")
                    delete_namespace("horizon")
                except Exception as e:
                    logger.warning(f"Could not clear Pinecone: {e}")

            # Reset state and invalidate cached check
            st.session_state.ingestion_done = False
            st.session_state.ingestion_log = []
            _check_stores_state.clear()
            st.rerun()

    # Data Explorer — always show when ingestion is done
    if st.session_state.ingestion_done:
        st.markdown("---")
        st.markdown("### Data Explorer")
        st.caption("Shows the actual values stored in Neon DB after ingestion.")

        try:
            from src.stores.sql_store import get_distinct_values, get_sample_rows

            distinct = get_distinct_values()
            if distinct:
                for client_id, info in distinct.items():
                    with st.expander(f"Client: **{client_id}** ({info['total_rows']} rows)", expanded=True):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown("**Products:**")
                            for p in info.get("products", []):
                                st.markdown(f"- `{p}`")
                        with col_b:
                            st.markdown("**Parameters:**")
                            for p in info.get("parameters", []):
                                st.markdown(f"- `{p}`")
                        with col_c:
                            st.markdown("**Regions:**")
                            for r in info.get("regions", []):
                                st.markdown(f"- `{r}`")

                        st.markdown("**Sample rows:**")
                        samples = get_sample_rows(client_id, limit=5)
                        if samples:
                            import pandas as pd
                            st.dataframe(pd.DataFrame(samples))
            else:
                st.info("No data found in the database.")
        except Exception as e:
            st.error(f"Could not query DB: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: CHAT
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Chat":
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

    # Example queries
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

    # Clear conversation button
    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.last_trace = None
            st.session_state.last_sources = None
            st.session_state.last_route_info = None
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle example query selection
    if "example_query" in st.session_state:
        user_input = st.session_state.pop("example_query")
    else:
        user_input = st.chat_input("Ask a question about product specifications...")

    # ── Query Processing ──
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()

            try:
                from langchain_core.messages import AIMessage, HumanMessage
                from src.graph.nodes import stream_synthesis, stream_conversational

                # Phase 1: Routing & Retrieval with live progress
                _NODE_LABELS = {
                    "query_analyzer": "Analyzing query...",
                    "pre_router": "Extracting signals...",
                    "llm_router": "Routing to retriever...",
                    "structured_retriever": "Querying database...",
                    "unstructured_retriever": "Searching documents...",
                    "hybrid_retriever": "Hybrid retrieval (SQL + vector)...",
                    "conversational_responder": "Responding...",
                }

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
                    # event is {node_name: state_update}
                    for node_name, state_update in event.items():
                        label = _NODE_LABELS.get(node_name, f"{node_name}...")
                        # Show progress with result info
                        detail = ""
                        if node_name == "llm_router" and state_update.get("route"):
                            detail = f" > `{state_update['route']}`"
                        elif node_name in ("structured_retriever", "hybrid_retriever"):
                            rows = len(state_update.get("sql_results", []))
                            vecs = len(state_update.get("vector_results", []))
                            if rows or vecs:
                                detail = f" > {rows} rows, {vecs} docs"
                        elif node_name == "unstructured_retriever":
                            vecs = len(state_update.get("vector_results", []))
                            detail = f" > {vecs} results"
                        elif node_name == "conversational_responder":
                            is_conversational = True

                        status_placeholder.markdown(f"*{label}{detail}*")
                        retrieval_result.update(state_update)

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

    # ── Information Panels ──
    if st.session_state.last_route_info or st.session_state.last_trace:
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
                    icon = step["icon"]
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
