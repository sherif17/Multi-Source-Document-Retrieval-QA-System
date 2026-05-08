"""
Pinecone vector store for unstructured document chunks.

Architecture decisions:
- Namespace-per-client for hard isolation (aurora/horizon namespaces)
- Metadata filtering as secondary isolation layer
- text-embedding-3-small (1536 dims, cosine similarity)
- Batch upsert for efficiency during ingestion
- Score threshold filtering to avoid low-relevance noise

Tradeoffs:
- Pinecone over ChromaDB: cloud-native, enables Render deployment, production-scalable
- Namespace isolation over metadata-only: physically separate data paths,
  queries to namespace "aurora" cannot return "horizon" vectors by design
- Cosine over dot-product: normalized embeddings, interpretable scores (0-1)
"""

from typing import Optional

import openai
from pinecone import Pinecone, ServerlessSpec

from src.config import settings
from src.models import ChunkMetadata, DocumentChunk
from src.utils.logger import logger

# ═══════════════════════════════════════════════════════════════════════════════
# CLIENT INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

_pc: Optional[Pinecone] = None
_index = None


def _get_pinecone() -> Pinecone:
    """Lazy-initialize Pinecone client."""
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc


def get_index():
    """Get or create the Pinecone index."""
    global _index
    if _index is not None:
        return _index

    pc = _get_pinecone()
    index_name = settings.pinecone_index_name

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        logger.info(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=settings.embedding_dimensions,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    _index = pc.Index(index_name)
    logger.info(f"Connected to Pinecone index: {index_name}")
    return _index


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts using OpenAI.

    Returns list of 1536-dimensional vectors.
    """
    if not texts:
        return []

    client = openai.OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    embeddings = [item.embedding for item in response.data]
    logger.debug(f"Generated {len(embeddings)} embeddings")
    return embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query string. Convenience wrapper."""
    return embed_texts([query])[0]


# ═══════════════════════════════════════════════════════════════════════════════
# WRITE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════


def upsert_chunks(chunks: list[DocumentChunk], namespace: str) -> int:
    """
    Embed and upsert document chunks into Pinecone.

    Uses namespace for client isolation. Each chunk gets:
    - A unique ID (deterministic, enables idempotent re-ingestion)
    - An embedding vector
    - Metadata for filtering and citation

    Returns number of vectors upserted.
    """
    if not chunks:
        return 0

    index = get_index()

    # Batch embed all texts
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts)

    # Prepare vectors for upsert
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        metadata = chunk.metadata.model_dump()
        # Pinecone metadata must be flat (no nested objects)
        # Convert list fields to comma-separated strings
        metadata["products_mentioned"] = ",".join(metadata["products_mentioned"])
        metadata["regions_mentioned"] = ",".join(metadata["regions_mentioned"])
        metadata["chunk_type"] = metadata["chunk_type"].value if hasattr(metadata["chunk_type"], "value") else metadata["chunk_type"]
        # Add text to metadata for retrieval
        metadata["text"] = chunk.text[:1000]  # Pinecone metadata limit

        vectors.append({
            "id": chunk.id,
            "values": embedding,
            "metadata": metadata,
        })

    # Batch upsert (Pinecone handles batches up to 100)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)

    logger.info(f"Upserted {len(vectors)} vectors to namespace '{namespace}'")
    return len(vectors)


# ═══════════════════════════════════════════════════════════════════════════════
# READ OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════


def search(
    query: str,
    namespace: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> list[dict]:
    """
    Semantic search within a client's namespace.

    Client isolation is enforced by namespace — physically impossible
    to return vectors from another client's namespace.

    Returns list of dicts with: text, score, metadata.
    """
    if top_k is None:
        top_k = settings.vector_top_k
    if score_threshold is None:
        score_threshold = settings.vector_score_threshold

    index = get_index()
    query_embedding = embed_query(query)

    results = index.query(
        vector=query_embedding,
        namespace=namespace,
        top_k=top_k,
        include_metadata=True,
    )

    # Filter by score threshold and format results
    filtered = []
    for match in results.matches:
        if match.score >= score_threshold:
            filtered.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "source_file": match.metadata.get("source_file", ""),
                "chunk_type": match.metadata.get("chunk_type", ""),
                "paragraph_index": match.metadata.get("paragraph_index", -1),
                "client_id": match.metadata.get("client_id", namespace),
            })

    # If all results are below threshold, take top 2 with low-confidence flag
    if not filtered and results.matches:
        for match in results.matches[:2]:
            filtered.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "source_file": match.metadata.get("source_file", ""),
                "chunk_type": match.metadata.get("chunk_type", ""),
                "paragraph_index": match.metadata.get("paragraph_index", -1),
                "client_id": match.metadata.get("client_id", namespace),
                "low_confidence": True,
            })

    logger.debug(
        f"Vector search in '{namespace}': {len(results.matches)} total, "
        f"{len(filtered)} above threshold ({score_threshold})"
    )
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════


def delete_namespace(namespace: str) -> None:
    """Delete all vectors in a namespace. Used for clean re-ingestion."""
    index = get_index()
    index.delete(delete_all=True, namespace=namespace)
    logger.info(f"Deleted all vectors in namespace '{namespace}'")


def get_index_stats() -> dict:
    """Get vector counts per namespace for observability."""
    index = get_index()
    stats = index.describe_index_stats()
    return {
        "total_vectors": stats.total_vector_count,
        "namespaces": {
            ns: info.vector_count
            for ns, info in (stats.namespaces or {}).items()
        },
    }
