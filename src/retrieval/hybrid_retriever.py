"""
Hybrid retriever: orchestrates both structured and unstructured retrieval.

Handles three scenarios:
1. hybrid_sql_primary: SQL leads, vector supplements with context
2. hybrid_vec_primary: Vector leads, SQL supplements with data
3. cross_client_hybrid: Separate isolated retrievals per client, merged at output

Critical design: Cross-client queries execute EACH client's retrieval in
complete isolation. Data is only "merged" in the result aggregation —
never at the query level. This prevents accidental data leakage between
client namespaces/partitions.
"""

from src.models import Source
from src.utils.logger import logger

from .structured_retriever import (
    build_sources_from_sql,
    format_sql_results_for_synthesis,
    retrieve_structured,
)
from .unstructured_retriever import (
    build_sources_from_vector,
    format_vector_results_for_synthesis,
    retrieve_unstructured,
)


def retrieve_hybrid(
    query: str,
    clients: list[str],
    strategy: str,
    sub_queries: list[str] | None = None,
    needs_narrative: bool = True,
) -> dict:
    """
    Execute hybrid retrieval across one or more clients.

    For cross-client queries, each client's data is retrieved independently
    to maintain strict isolation. Results are labeled with client_id so
    the synthesis LLM can attribute correctly.

    Args:
        query: The user's question (or reformulated query)
        clients: List of client IDs to query
        strategy: Route strategy (determines retrieval balance)
        sub_queries: Optional decomposed sub-queries for complex questions
        needs_narrative: Whether to include vector search

    Returns dict with:
        - sql_results: All SQL rows (labeled with client)
        - vector_results: All vector matches (labeled with client)
        - sql_query: Last SQL generated (for trace)
        - sql_error: Any SQL error encountered
        - sources: Combined Source objects for citation
        - formatted_sql: Human-readable SQL results
        - formatted_vector: Human-readable vector results
    """
    all_sql_results: list[dict] = []
    all_vector_results: list[dict] = []
    all_sources: list[Source] = []
    all_sql_queries: list[str] = []
    sql_error = None

    # For cross-client: use the main query per client (avoids redundant sub-query calls)
    # For single-client: use sub-queries if available for decomposed retrieval
    if strategy == "cross_client_hybrid":
        per_client_queries = [query]
    else:
        per_client_queries = sub_queries if sub_queries else [query]

    for client_id in clients:
        logger.info(f"Hybrid retrieval for client '{client_id}'")

        # ── Structured path ──
        if strategy in ("hybrid_sql_primary", "cross_client_hybrid", "structured_only"):
            for q in per_client_queries:
                results, sql_q, err = retrieve_structured(q, client_id)
                # Tag results with client for synthesis attribution
                for r in results:
                    r["_client_id"] = client_id
                all_sql_results.extend(results)
                if sql_q:
                    all_sql_queries.append(f"-- Client: {client_id}\n{sql_q}")
                if err:
                    sql_error = err

            all_sources.extend(build_sources_from_sql(all_sql_results))

        # ── Unstructured path ──
        if needs_narrative or strategy in (
            "hybrid_vec_primary",
            "hybrid_sql_primary",
            "cross_client_hybrid",
            "unstructured_only",
        ):
            vec_results = retrieve_unstructured(query, client_id)
            # Tag results with client
            for r in vec_results:
                r["_client_id"] = client_id
            all_vector_results.extend(vec_results)
            all_sources.extend(build_sources_from_vector(vec_results))

    # Format for synthesis LLM
    formatted_sql = _format_with_client_labels(all_sql_results, "sql")
    formatted_vector = _format_with_client_labels(all_vector_results, "vector")

    logger.info(
        f"Hybrid retrieval complete: {len(all_sql_results)} SQL rows, "
        f"{len(all_vector_results)} vector results across {len(clients)} client(s)"
    )

    return {
        "sql_results": all_sql_results,
        "vector_results": all_vector_results,
        "sql_query": "\n\n".join(all_sql_queries) if all_sql_queries else "",
        "sql_error": sql_error,
        "sources": all_sources,
        "formatted_sql": formatted_sql,
        "formatted_vector": formatted_vector,
    }


def _format_with_client_labels(results: list[dict], store_type: str) -> str:
    """Format results with client labels for cross-client distinction."""
    if not results:
        return f"No {store_type} data found."

    if store_type == "sql":
        lines = []
        for i, row in enumerate(results, 1):
            client = row.get("_client_id", "?")
            parts = [f"[Client: {client}]"]
            if row.get("product_name"):
                parts.append(f"Product: {row['product_name']}")
            if row.get("parameter"):
                parts.append(f"Parameter: {row['parameter']}")
            if row.get("value") is not None:
                parts.append(f"Value: {row['value']}")
            if row.get("unit"):
                parts.append(f"Unit: {row['unit']}")
            if row.get("region"):
                parts.append(f"Region: {row['region']}")
            if row.get("notes"):
                parts.append(f"Notes: {row['notes']}")
            if row.get("source_file"):
                parts.append(f"Source: {row['source_file']}")
            lines.append(f"[{i}] {' | '.join(parts)}")
        return "\n".join(lines)

    else:  # vector
        lines = []
        for i, r in enumerate(results, 1):
            client = r.get("_client_id", r.get("client_id", "?"))
            score = r.get("score", 0)
            text = r.get("text", "")
            source = r.get("source_file", "unknown")
            lines.append(
                f"[{i}] [Client: {client}] (score: {score:.3f}) "
                f"Source: {source}\n"
                f"    \"{text}\""
            )
        return "\n\n".join(lines)
