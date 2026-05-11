# Multi-Source Document Retrieval & QA System — Demo Walkthrough

## Table of Contents
- [High-Level Design](#high-level-design)
- [Architecture Overview](#architecture-overview)
- [Text Extraction & Ingestion Pipeline](#text-extraction--ingestion-pipeline)
- [Structured vs. Unstructured Separation](#structured-vs-unstructured-separation)
- [Client Separation & Data Isolation](#client-separation--data-isolation)
- [Query Routing](#query-routing)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Answers to the 5 Queries](#answers-to-the-5-queries)
- [Technology Stack](#technology-stack)

---

## High-Level Design

### Business Need

Paint industry clients (Aurora Paints, Horizon Coatings) maintain product safety documentation in **mixed formats** — spreadsheets with numeric compliance limits and Word documents with regulatory guidance. Compliance teams need to query across both data types using natural language:

- *"What's the maximum VOC limit for product X in the EU?"* → requires **exact numeric lookup**
- *"Where is enhanced ventilation recommended?"* → requires **semantic search over narrative text**
- *"Which client has stricter limits?"* → requires **cross-client comparison from both stores**

### Solution

A **dual-store retrieval system** with intelligent query routing that:

1. **Ingests** heterogeneous documents (XLSX + DOCX + embedded images)
2. **Separates** structured spec data → SQL database, narrative text → vector database
3. **Routes** each query to the optimal retrieval strategy
4. **Synthesizes** grounded answers with source citations
5. **Enforces** strict client data isolation at every layer

```mermaid
graph LR
    subgraph Input Sources
        A[XLSX Spreadsheets] 
        B[DOCX Documents]
        C[Embedded Images]
    end

    subgraph Dual-Store Architecture
        D[(PostgreSQL<br/>Neon DB)]
        E[(Pinecone<br/>Vector DB)]
    end

    subgraph Intelligent Retrieval
        F[Query Router]
        G[Answer Synthesis]
    end

    A -->|Schema Harmonization| D
    B -->|Classification| D
    B -->|Embedding| E
    C -->|GPT-4o Vision| D
    C -->|OCR Text| E

    F -->|Structured Query| D
    F -->|Semantic Search| E
    F -->|Hybrid| D & E
    D --> G
    E --> G
    G -->|Grounded Answer| H[User]
```

---

## Architecture Overview

```mermaid
graph TB
    subgraph "User Interface (Streamlit)"
        UI[Chat Interface]
        IP[Ingestion Panel]
    end

    subgraph "LangGraph Orchestration Engine"
        QA[Query Analyzer<br/><i>Reformulate with history</i>]
        PR[Pre-Router<br/><i>Deterministic signals</i>]
        LR[LLM Router<br/><i>GPT-4o-mini classification</i>]
        
        subgraph "Retrieval Nodes"
            SR[Structured Retriever<br/><i>Text-to-SQL</i>]
            UR[Unstructured Retriever<br/><i>Semantic search</i>]
            HR[Hybrid Retriever<br/><i>Combined retrieval</i>]
            CR[Conversational<br/><i>No retrieval needed</i>]
        end
        
        AS[Answer Synthesizer<br/><i>GPT-4.1 grounded generation</i>]
        GC[Guardrails Checker<br/><i>Isolation + hallucination</i>]
    end

    subgraph "Data Stores"
        SQL[(Neon PostgreSQL<br/>Structured specs)]
        VEC[(Pinecone<br/>Narrative embeddings)]
    end

    subgraph "Ingestion Pipeline"
        XL[XLSX Loader<br/><i>Schema detection</i>]
        DX[DOCX Parser<br/><i>Paragraph extraction</i>]
        CC[Content Classifier<br/><i>Rules + LLM fallback</i>]
        SE[Spec Extractor<br/><i>Regex + LLM fallback</i>]
        IE[Image Extractor<br/><i>GPT-4o Vision</i>]
    end

    UI --> QA --> PR --> LR
    LR -->|structured_only| SR
    LR -->|unstructured_only| UR
    LR -->|hybrid / cross_client| HR
    LR -->|conversational| CR

    SR --> AS
    UR --> AS
    HR --> AS
    CR --> UI

    AS --> GC
    GC -->|Pass| UI
    GC -->|Fail: Retry| HR

    SR --> SQL
    UR --> VEC
    HR --> SQL & VEC

    IP --> XL --> SQL
    IP --> DX --> CC
    CC -->|STRUCTURED| SE --> SQL
    CC -->|NARRATIVE| VEC
    IP --> IE --> SQL & VEC
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Dual-store** (SQL + Vector) | Numeric specs need exact queries (WHERE, MAX); narrative needs semantic search |
| **LangGraph** orchestration | Conditional routing + retry loops + state machine observability |
| **Two-tier routing** | Deterministic rules (free) → LLM classification (accurate) |
| **Namespace isolation** | Physically separate vector spaces per client (strongest possible) |
| **Tiered extraction** | Regex first (fast, free) → LLM fallback (handles edge cases) |

---

## Text Extraction & Ingestion Pipeline

### Overview

```mermaid
flowchart TB
    subgraph "Phase 1: XLSX Ingestion"
        direction TB
        X1[Read XLSX with pandas] --> X2[Detect Schema Variant<br/><i>Aurora vs Horizon fingerprinting</i>]
        X2 --> X3[Rename Columns<br/><i>Harmonize to canonical schema</i>]
        X3 --> X4[Pydantic Validation<br/><i>ProductSpec model</i>]
        X4 --> X5[INSERT into Neon DB]
    end

    subgraph "Phase 2: DOCX Ingestion"
        direction TB
        D1[Parse DOCX<br/><i>python-docx</i>] --> D2[Extract Paragraphs<br/><i>with style + image detection</i>]
        D2 --> D3[Classify Content<br/><i>Rules → LLM fallback</i>]
        D3 -->|STRUCTURED| D4[Extract Specs<br/><i>Regex → LLM fallback</i>]
        D3 -->|NARRATIVE| D5[Chunk & Embed<br/><i>text-embedding-3-small</i>]
        D3 -->|METADATA| D6[Extract Doc Context<br/><i>Client, product, variant</i>]
        D3 -->|IGNORE| D7[Skip<br/><i>Assignment instructions</i>]
        D4 --> D8[INSERT into Neon DB]
        D5 --> D9[Upsert to Pinecone<br/><i>namespace = client_id</i>]
    end

    subgraph "Phase 3: Image Extraction (Bonus)"
        direction TB
        I1[Extract Embedded Images<br/><i>from DOCX relationships</i>] --> I2[GPT-4o Vision<br/><i>Structured JSON extraction</i>]
        I2 --> I3[ProductSpec validation]
        I3 --> I4[INSERT into Neon DB]
        I2 --> I5[Format as text]
        I5 --> I6[Embed & Upsert to Pinecone]
    end

    %% Force vertical layout: phases stack top-to-bottom
    X5 ~~~ D1
    D8 ~~~ I1
```

### Schema Harmonization

Aurora and Horizon use **completely different column names** for identical data:

```mermaid
graph LR
    subgraph "Aurora Schema"
        A1[client_name]
        A2[product_name]
        A3[region]
        A4[parameter]
        A5[value]
        A6[unit]
    end

    subgraph "Canonical Schema"
        C1[client_id]
        C2[product_name]
        C3[region]
        C4[parameter]
        C5[value]
        C6[unit]
    end

    subgraph "Horizon Schema"
        H1[supplier_name]
        H2[product_line]
        H3[market]
        H4[metric]
        H5[metric_value]
        H6[metric_unit]
    end

    A1 --> C1
    A2 --> C2
    A3 --> C3
    A4 --> C4
    A5 --> C5
    A6 --> C6

    H1 --> C1
    H2 --> C2
    H3 --> C3
    H4 --> C4
    H5 --> C5
    H6 --> C6
```

**Implementation:** Fingerprint-based auto-detection identifies the schema variant from column names, then applies the correct rename mapping. Adding a new client = one new fingerprint entry.

### Content Classification Strategy

```mermaid
flowchart TD
    P[Paragraph] --> R1{Assignment instruction?}
    R1 -->|Yes| IGNORE[IGNORE]
    R1 -->|No| R2{Has embedded image?}
    R2 -->|Yes| IMAGE[IMAGE]
    R2 -->|No| R3{Heading or 'Client:' prefix?}
    R3 -->|Yes| META[METADATA]
    R3 -->|No| R4{Has numeric + unit<br/>AND spec keyword?}
    R4 -->|Yes| STRUCT[STRUCTURED<br/><i>→ SQL store + Vector store</i>]
    R4 -->|No| NARR[NARRATIVE<br/><i>→ Vector store only</i>]

    style STRUCT fill:#e1f5fe
    style NARR fill:#f3e5f5
    style IMAGE fill:#fff3e0
    style META fill:#e8f5e9
    style IGNORE fill:#fafafa
```

**Classification accuracy:** Rule-based classification handles ~95% of paragraphs deterministically. The remaining ~5% ambiguous cases fall through to an LLM classifier.

---

## Structured vs. Unstructured Separation

### What Goes Where

```mermaid
graph TB
    subgraph "Structured Store (Neon PostgreSQL)"
        direction TB
        S1["Numeric specs from XLSX"]
        S2["Numeric specs extracted from DOCX text"]
        S3["Numeric specs extracted from images (OCR)"]
    end

    subgraph "Unstructured Store (Pinecone Vector DB)"
        direction TB
        U1["Regulatory guidance paragraphs"]
        U2["Application recommendations"]
        U3["Policy and compliance text"]
    end

    %% Force vertical stacking
    S3 ~~~ U1

    style S1 fill:#e3f2fd
    style S2 fill:#e3f2fd
    style U1 fill:#f3e5f5
    style U2 fill:#f3e5f5
```

### Why Two Stores?

| Query Type | Best Store | Why |
|-----------|-----------|-----|
| "What is the max VOC for X?" | **SQL** | Exact filter + value lookup |
| "Maximum across 3 products?" | **SQL** | Aggregation (MAX, ORDER BY) |
| "What does guidance recommend?" | **Vector** | Semantic similarity search |
| "Summarize the policy for X" | **Vector** | Narrative comprehension |
| "Compare limits + explain context" | **Both** | Numeric comparison + narrative |

> **Key insight:** Embedding "30 g/L" as a vector loses its queryability. You can't do `MAX()` over embeddings. Conversely, stuffing narrative into SQL loses semantic richness.

---

## Client Separation & Data Isolation

### Three-Layer Defense Model

```mermaid
graph TB
    subgraph "Layer 1: Storage Isolation"
        direction LR
        NS1[Pinecone Namespace<br/>'aurora'] 
        NS2[Pinecone Namespace<br/>'horizon']
        SQL1["SQL: WHERE client_id = 'aurora'"]
        SQL2["SQL: WHERE client_id = 'horizon'"]
    end

    subgraph "Layer 2: Query Validation"
        direction LR
        V1[SQL Validator checks<br/>client_id filter exists]
        V2[Auto-injects filter<br/>if LLM forgot it]
        V3[Blocks queries referencing<br/>wrong client]
    end

    subgraph "Layer 3: Post-Generation Guardrails"
        direction LR
        G1[Scan answer for<br/>cross-client data leaks]
        G2[Retry if violation<br/>detected]
    end

    Q[User Query] --> NS1 & SQL1
    Q --> V1 --> V2 --> V3
    V3 --> G1 --> G2

    style NS1 fill:#c8e6c9
    style NS2 fill:#ffcdd2
```

### Isolation Mechanisms in Detail

| Layer | Mechanism | Strength |
|-------|-----------|----------|
| **Pinecone Namespaces** | Queries to namespace "aurora" *physically cannot* return "horizon" vectors | **Architectural** — impossible to bypass |
| **SQL WHERE clause** | Every query validated to contain `client_id = '{client}'` | **Enforced** — auto-injected if missing |
| **Wrong-client detection** | SQL referencing `'horizon'` when scoped to `'aurora'` raises error | **Proactive** — catches LLM mistakes |
| **Guardrails checker** | Post-synthesis scan for mentions of unauthorized clients | **Defense-in-depth** — catches hallucination |

### SQL Validation Pipeline (Security-Critical)

```mermaid
flowchart LR
    SQL[LLM-Generated SQL] --> C1{Starts with SELECT?}
    C1 -->|No| BLOCK1[❌ Block: Only SELECT allowed]
    C1 -->|Yes| C2{Forbidden keywords?<br/>DROP/DELETE/UPDATE/INSERT}
    C2 -->|Yes| BLOCK2[❌ Block: Forbidden keyword]
    C2 -->|No| C3{Multiple statements?}
    C3 -->|Yes| BLOCK3[❌ Block: No semicolons]
    C3 -->|No| C4{Has client_id filter?}
    C4 -->|No| FIX[⚠️ Auto-inject filter]
    C4 -->|Yes| C5{References wrong client?}
    C5 -->|Yes| BLOCK4[❌ Block: Wrong client]
    C5 -->|No| PASS[✅ Execute safely]
    FIX --> C5
```

---

## Query Routing

### Two-Tier Routing Architecture

```mermaid
flowchart TB
    Q[User Query] --> PR[Pre-Router<br/><i>Deterministic signal extraction</i>]
    
    PR --> |Signals| LR[LLM Router<br/><i>GPT-4o-mini intent classification</i>]

    subgraph "Pre-Router Signals (Free, Instant)"
        S1[extracted_clients: aurora/horizon/both]
        S2[has_numeric_ask: true/false]
        S3[has_narrative_ask: true/false]
        S4[is_cross_client: true/false]
        S5[has_aggregation: true/false]
    end

    LR --> R1[structured_only<br/><i>Text-to-SQL → Neon DB</i>]
    LR --> R2[unstructured_only<br/><i>Embed → Pinecone search</i>]
    LR --> R3[hybrid_sql_primary<br/><i>Both stores, SQL leads</i>]
    LR --> R4[cross_client_hybrid<br/><i>Per-client retrieval + merge</i>]
    LR --> R5[conversational<br/><i>No retrieval needed</i>]

    PR -.-> S1 & S2 & S3 & S4 & S5
    S1 & S2 & S3 & S4 & S5 -.-> LR

    style R1 fill:#e3f2fd
    style R2 fill:#f3e5f5
    style R3 fill:#fff9c4
    style R4 fill:#ffecb3
    style R5 fill:#e8f5e9
```

### Routing Decision Matrix

| Query Pattern | Pre-Router Signals | LLM Route Decision |
|--------------|-------------------|-------------------|
| Specific numeric value (limit, content) | `has_numeric_ask=True` | `structured_only` |
| Guidance, recommendations, policy | `has_narrative_ask=True` | `unstructured_only` |
| Numeric + contextual explanation | Both signals True | `hybrid_sql_primary` |
| Comparing two clients | `is_cross_client=True` | `cross_client_hybrid` |
| "Hello", "thanks", follow-up chat | Neither signal | `conversational` |

### Fallback Strategy

- **LLM confidence < 0.5** → Default to `hybrid_sql_primary` (over-retrieval is safer than under-retrieval)
- **LLM call fails** → Use pre-route signals heuristically
- **No data found** → Retry with broader retrieval strategy

---

## Retrieval Pipeline

### Structured Retrieval (Text-to-SQL)

```mermaid
sequenceDiagram
    participant Q as Query
    participant Gen as SQL Generator<br/>(GPT-4o-mini)
    participant Val as SQL Validator<br/>(5-layer check)
    participant DB as Neon PostgreSQL
    participant Retry as Broadening Retry

    Q->>Gen: Natural language query + client_id
    Note over Gen: Dynamic prompt includes<br/>actual DB values
    Gen->>Val: Generated SQL
    Val->>Val: SELECT only? ✓<br/>No forbidden keywords? ✓<br/>Has client_id filter? ✓
    Val->>DB: Validated SQL
    DB->>DB: Execute query
    
    alt Results found
        DB-->>Q: Row data + SQL trace
    else 0 rows returned
        DB->>Retry: Empty result
        Retry->>Gen: Broader prompt<br/>(wider ILIKE, drop region)
        Gen->>Val: New SQL
        Val->>DB: Execute
        DB-->>Q: Broader results
    end
```

### Unstructured Retrieval (Semantic Search)

```mermaid
sequenceDiagram
    participant Q as Query
    participant Emb as OpenAI Embeddings<br/>(text-embedding-3-small)
    participant PC as Pinecone<br/>(client namespace)
    participant Filter as Score Filter

    Q->>Emb: Embed query text
    Emb->>PC: Vector query (top_k=5)<br/>namespace = client_id
    PC->>Filter: Ranked results with scores
    
    alt Scores ≥ threshold (0.7)
        Filter-->>Q: Filtered results with metadata
    else All below threshold
        Filter-->>Q: Top 2 results + low_confidence flag
    end
```

### Hybrid Retrieval (Cross-Client)

```mermaid
sequenceDiagram
    participant Q as Query
    participant HR as Hybrid Retriever
    participant SQL as SQL Store
    participant Vec as Vector Store

    Q->>HR: Query + [aurora, horizon]
    
    loop For each client
        HR->>SQL: SQL retrieval (client-scoped)
        SQL-->>HR: Structured results
        HR->>Vec: Vector search (client namespace)
        Vec-->>HR: Narrative results
    end
    
    HR->>HR: Merge results with<br/>client labels for attribution
    HR-->>Q: Combined results<br/>(ready for comparative synthesis)
```

---

## Answers to the 5 Queries

### Query 1
> *"For client Aurora Paints, what is the maximum zinc content allowed in the finished EcoSafe Interior Wall Paint for the EU?"*

**Routing:** `structured_only` — numeric lookup, single client, specific product + region

**Retrieval:** Text-to-SQL generates:
```sql
SELECT product_name, parameter, value, unit 
FROM product_specs 
WHERE client_id = 'aurora' 
  AND product_name ILIKE '%Interior Wall%' 
  AND parameter ILIKE '%zinc%' 
  AND region = 'EU'
```

**Interpretation:** This is a direct specification lookup. The system queries the SQL store for Aurora's zinc content limit on their Interior Wall Paint in the EU market. The answer is a single numeric value with its unit, sourced from the structured product specs data.

---

### Query 2
> *"For client Aurora Paints, considering EcoSafe Ceiling Paint, EcoSafe Exterior Facade and EcoShield Floor Coating in the EU, what is the maximum internal VOC limit in g/L across these products?"*

**Routing:** `structured_only` — aggregation query, single client, multiple products

**Retrieval:** SQL with aggregation:
```sql
SELECT product_name, parameter, value, unit 
FROM product_specs 
WHERE client_id = 'aurora' 
  AND (product_name ILIKE '%Ceiling%' 
       OR product_name ILIKE '%Exterior%' 
       OR product_name ILIKE '%Floor%')
  AND parameter ILIKE '%voc%' 
  AND region = 'EU'
ORDER BY value DESC
```

**Interpretation:** The system identifies the MAX VOC value across 3 products by retrieving all matching rows and finding the highest value. This tests multi-product aggregation capability — the answer presents which product has the highest VOC limit and what that value is.

---

### Query 3
> *"According to the guidance for client Horizon Coatings' UltraSafe Interior Wall Paint, in which types of rooms is enhanced ventilation or longer airing-out periods specifically recommended?"*

**Routing:** `unstructured_only` — narrative/guidance question, semantic understanding needed

**Retrieval:** Semantic search in Pinecone namespace `horizon`, embedding the query and finding paragraphs about ventilation recommendations.

**Interpretation:** This query cannot be answered from SQL (no "room type" column exists). It requires understanding narrative guidance text. The system performs semantic search over Horizon's embedded paragraphs and synthesizes an answer listing specific room types mentioned in the safety guidance. The answer references the source DOCX document.

---

### Query 4
> *"Comparing client Aurora Paints and client Horizon Coatings, which client sets a stricter VOC limit for interior wall paint in the EU, and what additional usage guidance is mentioned for sensitive environments across their products?"*

**Routing:** `cross_client_hybrid` — cross-client comparison, numeric values + narrative guidance needed

**Retrieval:** Per-client hybrid retrieval:
- **SQL (Aurora):** VOC limit for Interior Wall Paint, EU → `30 g/L`
- **SQL (Horizon):** VOC limit for Interior Wall Paint, EU → `25 g/L`
- **Vector (both):** Semantic search for "sensitive environments usage guidance" → guidance paragraphs from both clients

**Interpretation:** This is a **cross-client hybrid query** — it needs both numeric comparison AND qualitative narrative. The system retrieves VOC limits from SQL for each client independently (maintaining isolation), then searches both vector namespaces for guidance on sensitive environments. The synthesis combines the numeric comparison with relevant usage recommendations from both clients' documents.

---

### Query 5
> *"For client Aurora Paints, what internal VOC limit in g/L is set for EcoSafe Kitchen & Bath in the EU for typical residential projects?"*

**Routing:** `structured_only` or `hybrid_sql_primary` — primarily a numeric lookup, but "typical residential projects" may trigger hybrid for context

**Retrieval:**
1. **SQL:** VOC limit for Kitchen & Bath, EU → numeric value (e.g., 32 g/L)
2. **Vector (if hybrid):** Semantic search for "residential projects Kitchen Bath" → contextual paragraphs

**Interpretation:** This query asks for a specific numeric value with a contextual qualifier ("typical residential projects"). The router may choose structured-only if it interprets this as a pure numeric lookup, or hybrid if the qualifier suggests narrative context is needed. Either way, the SQL store provides the definitive answer. If hybrid is triggered, the vector store may surface additional context about residential application scenarios.

---

## Technology Stack

```mermaid
graph TB
    subgraph "Frontend"
        ST[Streamlit<br/><i>Chat UI + Ingestion Panel</i>]
    end

    subgraph "Orchestration"
        LG[LangGraph<br/><i>State machine workflow</i>]
    end

    subgraph "LLM Layer"
        M1[GPT-4o-mini<br/><i>Routing + SQL generation</i>]
        M2[GPT-4.1<br/><i>Answer synthesis</i>]
        M3[GPT-4o Vision<br/><i>Image extraction</i>]
        M4[text-embedding-3-small<br/><i>1536-dim embeddings</i>]
    end

    subgraph "Data Stores"
        NDB[(Neon PostgreSQL<br/><i>Serverless, cloud-native</i>)]
        PC[(Pinecone<br/><i>Serverless vector DB</i>)]
    end

    subgraph "Ingestion"
        PD[pandas + openpyxl<br/><i>XLSX processing</i>]
        DX[python-docx<br/><i>DOCX parsing</i>]
        PY[Pydantic<br/><i>Data validation</i>]
    end

    subgraph "Deployment"
        RN[Render.com<br/><i>Docker container</i>]
    end

    ST --> LG
    LG --> M1 & M2
    LG --> NDB & PC
    M4 --> PC
    PD & DX --> PY --> NDB & PC
    M3 --> NDB & PC
```

### Model Selection Rationale

| Model | Use Case | Why |
|-------|----------|-----|
| **GPT-4o-mini** | Routing, SQL generation, classification | Fast, cheap (~$0.001/call), sufficient for structured tasks |
| **GPT-4.1** | Answer synthesis | Higher quality reasoning for grounded, cited answers |
| **GPT-4o Vision** | Image spec extraction | Best-in-class multimodal understanding |
| **text-embedding-3-small** | Document + query embedding | Good quality at 1536 dims, cost-effective |

### Infrastructure

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **SQL Database** | Neon (serverless PostgreSQL) | Cloud-native, no server management, connection pooling |
| **Vector Database** | Pinecone (serverless) | Namespace isolation, auto-scaling, no infra management |
| **Deployment** | Render.com + Docker | Simple deployment, auto-scaling, HTTPS |
| **UI Framework** | Streamlit | Rapid prototyping, built-in chat components |

---

## Summary: Key Engineering Highlights

1. **Intelligent Routing** — Two-tier (rules + LLM) routing picks the optimal retrieval path per query
2. **Defense-in-Depth Security** — 5-layer SQL validation + namespace isolation + guardrails
3. **Self-Healing Retrieval** — Broadening retry on 0 results; guardrails failure triggers re-retrieval
4. **Schema Harmonization** — Auto-detects and normalizes different client data formats
5. **Tiered Extraction** — Regex (free, fast) → LLM (accurate, costly) at every classification stage
6. **Full Observability** — Retrieval trace captures per-node execution for debugging
7. **Production-Grade Design** — Idempotent ingestion, error isolation, graceful degradation
