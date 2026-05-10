"""
Session state initialization and startup checks.

Handles:
- Streamlit session state defaults
- Auto-detection of existing data in Neon DB + Pinecone
- Pre-compilation of the LangGraph workflow
"""

import streamlit as st

from src.utils.logger import logger


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED STARTUP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


@st.cache_resource(show_spinner=False)
def init_workflow():
    """Pre-compile LangGraph retrieval workflow once at startup."""
    from src.graph.workflow import get_retrieval_workflow
    return get_retrieval_workflow()


@st.cache_resource(show_spinner=False)
def check_stores_state() -> dict:
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
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def init_session_state():
    """Initialize all session state keys with defaults."""
    defaults = {
        "messages": [],
        "chat_history": [],
        "last_trace": None,
        "last_sources": None,
        "last_route_info": None,
        "ingestion_done": False,
        "ingestion_log": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Auto-detect existing data on startup
    if not st.session_state.ingestion_done:
        stores_state = check_stores_state()
        if stores_state["has_data"]:
            st.session_state.ingestion_done = True
            st.session_state["_auto_detected"] = stores_state
            logger.info(
                f"Auto-detected {stores_state['total_rows']} DB rows + "
                f"{stores_state['total_vectors']} vectors — skipping upload"
            )
