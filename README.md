# Multi-Source Document Retrieval & QA System

A production-grade intelligent document retrieval system that ingests XLSX and DOCX files for multiple clients, separates structured from unstructured information, routes queries to the optimal retrieval strategy, and generates grounded answers with citations — all while enforcing strict client data isolation.

---

## Architecture

```
┌──────────────────── Streamlit UI ────────────────────┐
│  Chat │ Routing Panel │ Trace Panel │ Sources Panel  │
└────────────────────────┬─────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  LangGraph Engine   │
              │  (8-node workflow)  │
              └──────────┬──────────┘
                ┌────────┼────────┐
                ▼        ▼        ▼
           ┌────────┐ ┌───────┐ ┌────────┐
           │Neon DB │ │Pinecone│ │ OpenAI │
           │(SQL)   │ │(Vector)│ │ (LLMs) │
           └────────┘ └───────┘ └────────┘
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| **LangGraph orchestration** | Explicit state machine — observable, debuggable, retry-capable |
| **Tiered routing** (rules + LLM) | Rules handle deterministic signals for free; LLM handles ambiguity |
| **3-layer client isolation** | Namespace isolation + SQL WHERE clause + guardrails check |
| **Separate models per task** | gpt-4o-mini for routing/SQL (cheap), gpt-4.1 for synthesis (quality) |
| **Dual OCR pipeline** | GPT-4o Vision (accurate) + Tesseract (free validation) |
| **Paragraph-level chunking** | Documents are short (~10 paras); paragraph = perfect semantic unit |

---

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd multi-source-doc-retrieval
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY=sk-...
#   NEON_DATABASE_URL=postgresql://...
#   PINECONE_API_KEY=...

# 3. Run ingestion (populates both stores)
python ingest.py

# 4. Launch the app
streamlit run app.py
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Orchestration | LangGraph | State machine workflow with conditional routing |
| Structured Store | Neon DB (PostgreSQL) | Product specifications, SQL-queryable |
| Vector Store | Pinecone (Serverless) | Narrative text, semantic search |
| LLM (Routing/SQL) | GPT-4o-mini | Cost-efficient classification and SQL generation |
| LLM (Synthesis) | GPT-4.1 | High-quality grounded answer generation |
| LLM (Vision) | GPT-4o | OCR/table extraction from embedded images |
| Embeddings | text-embedding-3-small | 1536-dim vectors for semantic search |
| UI | Streamlit | Chat interface with expandable panels |
| Deployment | Render.com (Docker) | Cloud-hosted, shareable URL |

---

## Project Structure

```
├── app.py                    # Streamlit entry point
├── ingest.py                 # CLI: run ingestion pipeline
├── src/
│   ├── config.py             # Pydantic Settings (env-based config)
│   ├── models.py             # All Pydantic data models
│   ├── ingestion/            # Data ingestion pipeline
│   │   ├── pipeline.py       # Orchestrator
│   │   ├── xlsx_loader.py    # Schema detection + normalization
│   │   ├── docx_loader.py    # Paragraph extraction
│   │   ├── content_classifier.py  # Structured vs narrative
│   │   ├── spec_extractor.py # Regex + LLM extraction
│   │   └── image_extractor.py # OCR/Vision pipeline
│   ├── stores/               # Database abstraction
│   │   ├── sql_store.py      # Neon DB operations
│   │   └── vector_store.py   # Pinecone operations
│   ├── retrieval/            # Query processing
│   │   ├── router.py         # Pre-router + LLM router
│   │   ├── structured_retriever.py  # Text-to-SQL
│   │   ├── unstructured_retriever.py  # Vector search
│   │   └── hybrid_retriever.py  # Combined retrieval
│   ├── graph/                # LangGraph orchestration
│   │   ├── state.py          # GraphState TypedDict
│   │   ├── nodes.py          # 8 node implementations
│   │   └── workflow.py       # Graph construction
│   └── utils/
│       ├── prompts.py        # All LLM prompts
│       ├── citations.py      # Source formatting
│       └── logger.py         # Logging config
├── tests/                    # Test suite
├── evaluation/               # Automated evaluation
├── docs/                     # Input data files
├── Dockerfile
└── render.yaml
```

---

## Query Routing

The system handles 5 types of queries:

| Type | Route | Example |
|------|-------|---------|
| **Numeric lookup** | SQL only | "What is the max zinc content for Aurora...?" |
| **Aggregation** | SQL only | "Maximum VOC across these 3 products?" |
| **Narrative** | Vector only | "In which rooms is ventilation recommended?" |
| **Hybrid** | SQL + Vector | "VOC limit for typical residential projects?" |
| **Cross-client** | Both stores × both clients | "Which client has stricter limits?" |

---

## Client Isolation

Three defense layers prevent data leakage:

1. **Pinecone namespaces** — physically separate vector spaces per client
2. **SQL WHERE clause** — mandatory `client_id` filter (validated post-generation)
3. **Guardrails checker** — post-synthesis scan for cross-client mentions

---

## Testing

```bash
# Run all tests (no API calls needed for unit tests)
pytest tests/ -v

# Run specific test files
pytest tests/test_ingestion.py -v
pytest tests/test_router.py -v
pytest tests/test_isolation.py -v

# Run evaluation (requires API keys + populated stores)
python -m evaluation.evaluate
```

---

## Deployment

The app deploys to Render.com via Docker:

1. Push to GitHub
2. Connect repo in Render dashboard
3. Set environment variables (API keys)
4. Auto-deploys on push to main

---

## License

Built for PwC Senior AI Engineer assessment.
