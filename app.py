"""
Streamlit application for the Multi-Source Document Retrieval system.

Two-page layout:
1. Upload & Ingest — Upload XLSX/DOCX files and watch real-time ingestion
2. Chat — Query the ingested data with full observability

Architecture: Thin UI orchestrator — page logic lives in ui/ package,
all intelligence lives in src/graph/workflow.py.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from ui.state import init_session_state, init_workflow
from ui.sidebar import render_sidebar
from ui.ingest_page import render_ingest_page
from ui.chat_page import render_chat_page

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
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

init_session_state()
workflow = init_workflow()

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

page = render_sidebar()

if page == "Upload & Ingest":
    render_ingest_page()
elif page == "Chat":
    render_chat_page(workflow)
