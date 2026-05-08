"""
Unstructured retriever: semantic search over Pinecone vector store.

Client isolation is enforced at the namespace level — queries to
namespace "aurora" physically cannot return "horizon" vectors.
This is the strongest possible isolation (not just a filter).

Design decisions:
- No LLM call needed (just embedding + search) — fast and cheap
- Score threshold filtering removes low-relevance noise
- Fallback: if all results below threshold, take top 2 with low-confidence flag
- Results include full metadata for citation propagation
"""

from src.config import settings
from src.models import Source
from src.stores.vector_store import search as vector_search
from src.utils.logger import logger


def retrieve_unstructured(
    query: str,
    client_id: str,
    top_k: int | None = None,
) -> list[dict]:
    """
    Semantic search within a client's vector namespace.

    Args:
        query: The search query (will be embedded)
        client_id: Client namespace to search within
        top_k: Number of results (default from settings)

    Returns:
        List of result dicts with: text, score, source_file, chunk_type, etc.
    """
    results = vector_search(
        query=query,
        namespace=client_id,
        top_k=top_k or settings.vector_top_k,
        score_threshold=settings.vector_score_threshold,
    )

    logger.info(
        f"Unstructured retrieval for client '{client_id}': "
        f"{len(results)} results returned"
    )

    return results


def format_vector_results_for_synthesis(results: list[dict]) -> str:
    """
    Format vector search results as readable text for the synthesis LLM.

    Includes relevance scores so the synthesizer can weigh sources.
    """
    if not results:
        return "No unstructured/narrative data found."

    lines = []
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        text = r.get("text", "")
        source = r.get("source_file", "unknown")
        chunk_type = r.get("chunk_type", "unknown")
        para_idx = r.get("paragraph_index", "?")

        lines.append(
            f"[{i}] (score: {score:.3f}, type: {chunk_type}) "
            f"Source: {source} (para {para_idx})\n"
            f"    \"{text}\""
        )

    return "\n\n".join(lines)


def build_sources_from_vector(results: list[dict]) -> list[Source]:
    """Convert vector results into Source objects for citation tracking."""
    sources = []
    for r in results:
        sources.append(
            Source(
                store="vector",
                content=r.get("text", "")[:200],
                file=r.get("source_file", "unknown"),
                detail=f"Paragraph {r.get('paragraph_index', '?')} ({r.get('chunk_type', 'unknown')})",
                relevance_score=r.get("score"),
            )
        )
    return sources
