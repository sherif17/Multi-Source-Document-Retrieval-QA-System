"""
Sidebar rendering — navigation, architecture info, and data status.
"""

import streamlit as st


def render_sidebar() -> str:
    """Render the sidebar and return the selected page name."""
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
            "- GPT-4o Vision (images)"
        )

        st.markdown("---")
        status_label = "Ingested" if st.session_state.ingestion_done else "Not ingested"
        st.markdown(f"**Data Status:** {status_label}")

    return page
