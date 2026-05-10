"""
Upload & Ingest page — file upload, ingestion pipeline, and data explorer.
"""

import tempfile
from pathlib import Path

import streamlit as st

from src.utils.logger import logger
from ui.state import check_stores_state


def render_ingest_page():
    """Render the Upload & Ingest page."""
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
        _show_uploaded_summary(uploaded_files)
        _run_ingestion(uploaded_files)

    elif st.session_state.ingestion_done:
        _show_previous_ingestion()

    # Data Explorer — always show when ingestion is done
    if st.session_state.ingestion_done:
        _render_data_explorer()


def _show_uploaded_summary(uploaded_files):
    """Display uploaded file metrics."""
    st.markdown("### Uploaded Files")
    cols = st.columns(len(uploaded_files))
    for i, f in enumerate(uploaded_files):
        with cols[i]:
            ftype = "[XLSX]" if f.name.endswith(".xlsx") else "[DOCX]"
            st.metric(label=f"{ftype} {f.name}", value=f"{f.size / 1024:.1f} KB")
    st.markdown("---")


def _run_ingestion(uploaded_files):
    """Run the ingestion pipeline with real-time progress."""
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


def _show_previous_ingestion():
    """Show summary of previous ingestion and re-ingest option."""
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
        check_stores_state.clear()
        st.rerun()


def _render_data_explorer():
    """Show the Data Explorer with DB contents."""
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
