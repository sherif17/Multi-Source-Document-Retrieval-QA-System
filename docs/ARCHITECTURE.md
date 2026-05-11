# System Architecture — Detailed Visual Reference

## Complete Data Flow

```mermaid
flowchart TB
    %% ─── INGESTION SIDE ───
    subgraph INGESTION["📥 Ingestion Pipeline"]
        direction TB
        
        subgraph FILES["Input Files"]
            XLSX1[aurora_specs.xlsx]
            XLSX2[horizon_specs.xlsx]
            DOCX1[aurora_safety_brief.docx]
            DOCX2[horizon_safety_brief.docx]
        end

        subgraph XLSX_PIPE["XLSX Processing"]
            SD[Schema Detection<br/><i>Fingerprint matching</i>]
            SH[Schema Harmonization<br/><i>Column rename → canonical</i>]
            PV[Pydantic Validation<br/><i>Type checking + normalization</i>]
        end

        subgraph DOCX_PIPE["DOCX Processing"]
            DP[Paragraph Extraction<br/><i>python-docx parser</i>]
            CL[Content Classifier<br/><i>5-type classification</i>]
            
            subgraph STRUCT_PATH["Structured Path"]
                RE[Regex Extraction<br/><i>VOC, lead, zinc, drying</i>]
                LE[LLM Extraction<br/><i>Fallback for edge cases</i>]
            end
            
            subgraph NARR_PATH["Narrative Path"]
                CH[Paragraph Chunking<br/><i>Natural semantic units</i>]
                EM[OpenAI Embedding<br/><i>1536 dimensions</i>]
            end
        end

        subgraph IMG_PIPE["Image Processing"]
            IX[Image Extraction<br/><i>From DOCX relationships</i>]
            VIS[GPT-4o Vision<br/><i>Structured JSON output</i>]
        end
    end

    %% ─── STORAGE ───
    subgraph STORAGE["💾 Dual-Store Architecture"]
        direction LR
        
        subgraph SQL_STORE["Neon PostgreSQL"]
            T1[("product_specs table<br/>─────────────────<br/>client_id | product_name<br/>region | parameter<br/>value | unit<br/>limit_type | notes<br/>source_file | source_type")]
        end
        
        subgraph VEC_STORE["Pinecone Vector DB"]
            N1[("Namespace: aurora<br/>─────────────────<br/>Regulatory guidance<br/>Application notes<br/>Policy paragraphs")]
            N2[("Namespace: horizon<br/>─────────────────<br/>Regulatory guidance<br/>Application notes<br/>Policy paragraphs")]
        end
    end

    %% ─── CONNECTIONS ───
    XLSX1 & XLSX2 --> SD --> SH --> PV --> T1
    DOCX1 & DOCX2 --> DP --> CL
    CL -->|STRUCTURED| RE
    RE -->|"~90%"| T1
    RE -->|"~10% fallback"| LE --> T1
    CL -->|NARRATIVE| CH --> EM --> N1 & N2
    DOCX1 & DOCX2 --> IX --> VIS --> T1
    VIS --> EM
```

---

## Query Processing Flow

```mermaid
flowchart TB
    %% ─── RETRIEVAL SIDE ───
    subgraph RETRIEVAL["🔍 Retrieval Pipeline (LangGraph)"]
        direction TB
        
        USER[/"User Query"/]
        
        subgraph ANALYSIS["Step 1: Query Understanding"]
            QA[Query Analyzer<br/><i>Reformulate with chat history</i>]
        end
        
        subgraph ROUTING["Step 2: Two-Tier Routing"]
            PR[Pre-Router<br/><i>Regex signal extraction</i>]
            LLM_R[LLM Router<br/><i>GPT-4o-mini classification</i>]
        end
        
        subgraph RETRIEVE["Step 3: Retrieval Execution"]
            SR["Structured Retriever<br/><i>Generate SQL → Validate → Execute</i>"]
            UR["Unstructured Retriever<br/><i>Embed → Search namespace</i>"]
            HR["Hybrid Retriever<br/><i>Per-client: SQL + Vector</i>"]
            CONV["Conversational<br/><i>Direct LLM response</i>"]
        end
        
        subgraph SYNTHESIS["Step 4: Answer Generation"]
            AS[Answer Synthesizer<br/><i>GPT-4.1 grounded generation</i>]
            GC[Guardrails Checker<br/><i>Isolation + hallucination detection</i>]
        end
        
        ANSWER[/"Final Answer with Citations"/]
    end

    %% ─── CONNECTIONS ───
    USER --> QA --> PR --> LLM_R
    LLM_R -->|"intent: lookup<br/>strategy: structured_only"| SR
    LLM_R -->|"intent: narrative<br/>strategy: unstructured_only"| UR
    LLM_R -->|"intent: comparison<br/>strategy: cross_client_hybrid"| HR
    LLM_R -->|"intent: conversational<br/>strategy: conversational"| CONV
    
    SR --> AS
    UR --> AS
    HR --> AS
    CONV --> ANSWER
    
    AS --> GC
    GC -->|"✅ Pass"| ANSWER
    GC -->|"❌ Fail: Retry"| HR
```

---

## Client Isolation Architecture

```mermaid
graph TB
    subgraph "Query: Aurora Client"
        Q1[User asks about Aurora]
        
        subgraph "Layer 1: Routing"
            R1["Pre-router detects 'aurora'<br/>→ Scopes all downstream ops"]
        end
        
        subgraph "Layer 2: Vector Isolation"
            V1["Pinecone: query ONLY<br/>namespace='aurora'<br/><i>Cannot access horizon data</i>"]
        end
        
        subgraph "Layer 3: SQL Isolation"
            S1["SQL Validator ensures:<br/>WHERE client_id = 'aurora'<br/><i>Auto-injected if missing</i>"]
        end
        
        subgraph "Layer 4: Post-Generation"
            G1["Guardrails scan answer<br/>for 'horizon' references<br/><i>Retry if found</i>"]
        end
    end

    Q1 --> R1 --> V1 & S1
    V1 & S1 --> G1
    G1 -->|Clean| ANS[Answer delivered]
    G1 -->|Violation| RETRY[Retry with stricter prompt]
```

---

## Retry & Self-Healing Mechanism

```mermaid
stateDiagram-v2
    [*] --> QueryAnalysis
    QueryAnalysis --> Routing
    Routing --> Retrieval
    
    Retrieval --> Synthesis: Results found
    Retrieval --> BroadenRetry: 0 results
    
    BroadenRetry --> Synthesis: Broader results found
    BroadenRetry --> Synthesis: Still empty (report "no data")
    
    Synthesis --> Guardrails
    
    Guardrails --> Done: All checks pass ✅
    Guardrails --> HybridRetry: Isolation violation ❌
    Guardrails --> HybridRetry: Suspected hallucination ❌
    
    HybridRetry --> Synthesis: Re-retrieve + re-synthesize
    
    state HybridRetry {
        [*] --> WidenRetrieval
        WidenRetrieval --> ReSynthesize
        ReSynthesize --> [*]
    }
    
    Done --> [*]
```

---

## Module Dependency Map

```mermaid
graph LR
    subgraph "UI Layer"
        CP[chat_page.py]
        IP[ingest_page.py]
    end

    subgraph "Orchestration"
        WF[workflow.py]
        ND[nodes.py]
    end

    subgraph "Retrieval"
        RT[router.py]
        SR[structured_retriever.py]
        UR[unstructured_retriever.py]
        HR[hybrid_retriever.py]
    end

    subgraph "Ingestion"
        PL[pipeline.py]
        XL[xlsx_loader.py]
        DL[docx_loader.py]
        CC[content_classifier.py]
        SE[spec_extractor.py]
        IE[image_extractor.py]
    end

    subgraph "Stores"
        SS[sql_store.py]
        VS[vector_store.py]
    end

    subgraph "Core"
        CF[config.py]
        MD[models/]
        PM[prompts/]
    end

    CP --> WF --> ND
    ND --> RT & SR & UR & HR
    SR --> SS
    UR --> VS
    HR --> SR & UR
    
    IP --> PL
    PL --> XL & DL & CC & SE & IE
    XL --> SS
    SE --> SS
    DL --> CC
    IE --> SS & VS
    CC --> VS
    
    RT --> PM
    SR --> PM & SS
    ND --> PM
    
    SS & VS --> CF
    MD --> CF
```

---

## Performance Characteristics

| Component | Latency | Cost per Query |
|-----------|---------|---------------|
| Pre-router (regex) | <1ms | Free |
| LLM Router (4o-mini) | ~1.5s | ~$0.001 |
| SQL Generation (4o-mini) | ~1.5s | ~$0.001 |
| SQL Execution (Neon) | ~100ms | Free tier |
| Vector Search (Pinecone) | ~200ms | Free tier |
| Answer Synthesis (4.1) | ~3s | ~$0.01 |
| **Total (structured)** | **~6s** | **~$0.012** |
| **Total (hybrid)** | **~7s** | **~$0.013** |
