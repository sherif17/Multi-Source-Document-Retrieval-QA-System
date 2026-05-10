"""
LangGraph node implementations — the 8 processing steps of the orchestration workflow.

Each node is a pure function: GraphState → partial GraphState update.
Nodes are composable, testable in isolation, and fully observable.

Node execution order:
1. query_analyzer → reformulate with chat history
2. pre_router → extract deterministic signals
3. llm_router → classify intent, pick strategy
4. structured_retriever → Text-to-SQL
5. unstructured_retriever → Pinecone search
6. hybrid_retriever → combined retrieval
7. answer_synthesizer → generate grounded answer (gpt-4.1)
8. guardrails_checker → validate quality, enforce isolation
"""

import time
from typing import Any

import openai

from src.config import settings
from src.retrieval.hybrid_retriever import retrieve_hybrid
from src.retrieval.router import pre_route, llm_route
from src.retrieval.structured_retriever import (
    build_sources_from_sql,
    format_sql_results_for_synthesis,
    retrieve_structured,
)
from src.retrieval.unstructured_retriever import (
    build_sources_from_vector,
    format_vector_results_for_synthesis,
    retrieve_unstructured,
)
from src.utils.logger import logger
from src.prompts import (
    QUERY_ANALYZER_SYSTEM,
    QUERY_ANALYZER_USER,
    SYNTHESIS_RETRY_SUFFIX,
    SYNTHESIS_SYSTEM,
    SYNTHESIS_USER,
)

from .state import GraphState


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1: QUERY ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════


def query_analyzer(state: GraphState) -> dict[str, Any]:
    """
    Reformulate the query using conversation history to resolve references.

    If no chat history exists, returns the query unchanged.
    Uses gpt-4o-mini for cost efficiency (reformulation is simple).
    """
    start = time.time()
    query = state["query"]
    chat_history = state.get("chat_history", [])

    # Skip reformulation if no history (saves an LLM call)
    if not chat_history:
        duration = (time.time() - start) * 1000
        return {
            "reformulated_query": query,
            "retrieval_trace": state.get("retrieval_trace", []) + [
                {
                    "node": "query_analyzer",
                    "action": "Query is self-contained (no history)",
                    "duration_ms": duration,
                    "result_count": 0,
                    "detail": None,
                }
            ],
        }

    # Format chat history for prompt
    history_text = _format_chat_history(chat_history)

    try:
        client = openai.OpenAI(api_key=settings.openai_api_key, timeout=15.0)
        response = client.chat.completions.create(
            model=settings.routing_model,
            messages=[
                {"role": "system", "content": QUERY_ANALYZER_SYSTEM},
                {
                    "role": "user",
                    "content": QUERY_ANALYZER_USER.format(
                        chat_history=history_text, query=query
                    ),
                },
            ],
            temperature=0,
            max_tokens=200,
        )
        reformulated = response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Query reformulation failed: {e}. Using original query.")
        reformulated = query

    duration = (time.time() - start) * 1000
    changed = reformulated != query

    logger.info(
        f"Query analyzer: {'reformulated' if changed else 'unchanged'} "
        f"({duration:.0f}ms)"
    )
    if changed:
        logger.debug(f"  Original: {query}")
        logger.debug(f"  Reformulated: {reformulated}")

    return {
        "reformulated_query": reformulated,
        "retrieval_trace": state.get("retrieval_trace", []) + [
            {
                "node": "query_analyzer",
                "action": f"Reformulated: {reformulated}" if changed else "Query unchanged",
                "duration_ms": duration,
                "result_count": 0,
                "detail": f"Original: {query}" if changed else None,
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 2: PRE-ROUTER
# ═══════════════════════════════════════════════════════════════════════════════


def pre_router(state: GraphState) -> dict[str, Any]:
    """
    Extract deterministic signals from the query (no LLM, pure regex).

    Fast, free, 100% reliable for:
    - Client detection
    - Cross-client detection
    - Keyword signal extraction
    """
    start = time.time()
    query = state["reformulated_query"]

    signals = pre_route(query)
    duration = (time.time() - start) * 1000

    return {
        "extracted_clients": signals.extracted_clients,
        "pre_route_signals": signals.model_dump(),
        "retrieval_trace": state.get("retrieval_trace", []) + [
            {
                "node": "pre_router",
                "action": (
                    f"Clients: {signals.extracted_clients} | "
                    f"numeric={'yes' if signals.has_numeric_ask else 'no'} "
                    f"narrative={'yes' if signals.has_narrative_ask else 'no'} "
                    f"cross_client={'yes' if signals.is_cross_client else 'no'}"
                ),
                "duration_ms": duration,
                "result_count": 0,
                "detail": None,
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3: LLM ROUTER
# ═══════════════════════════════════════════════════════════════════════════════


def llm_router(state: GraphState) -> dict[str, Any]:
    """
    Deep intent classification and route selection using gpt-4o-mini.

    Takes pre-route signals as hints, returns typed RouteDecision with
    intent, strategy, confidence, and optional sub-query decomposition.
    """
    start = time.time()
    query = state["reformulated_query"]

    from src.models import PreRouteSignals
    signals = PreRouteSignals(**state["pre_route_signals"])

    decision = llm_route(query, signals)
    duration = (time.time() - start) * 1000

    return {
        "route_decision": decision.model_dump(),
        "route": decision.strategy,
        "route_reasoning": decision.reasoning,
        "route_confidence": decision.confidence,
        "sub_queries": decision.sub_queries,
        "retrieval_trace": state.get("retrieval_trace", []) + [
            {
                "node": "llm_router",
                "action": (
                    f"Route: {decision.strategy} | "
                    f"Intent: {decision.intent} | "
                    f"Confidence: {decision.confidence:.2f}"
                ),
                "duration_ms": duration,
                "result_count": 0,
                "detail": decision.reasoning,
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4: STRUCTURED RETRIEVER
# ═══════════════════════════════════════════════════════════════════════════════


def structured_retriever(state: GraphState) -> dict[str, Any]:
    """
    Text-to-SQL retrieval from Neon DB.

    Generates SQL, validates for safety, executes, returns results.
    Only queries the first client in extracted_clients (single-client route).
    """
    start = time.time()
    query = state["reformulated_query"]
    client_id = state["extracted_clients"][0]

    results, sql_query, error = retrieve_structured(query, client_id)
    duration = (time.time() - start) * 1000

    sources = [s.model_dump() for s in build_sources_from_sql(results)]

    return {
        "sql_query": sql_query,
        "sql_results": results,
        "sql_error": error,
        "vector_results": [],
        "formatted_sql": format_sql_results_for_synthesis(results),
        "formatted_vector": "No unstructured data retrieved (structured-only route).",
        "sources": sources,
        "retrieval_trace": state.get("retrieval_trace", []) + [
            {
                "node": "structured_retriever",
                "action": f"SQL execution → {len(results)} rows",
                "duration_ms": duration,
                "result_count": len(results),
                "detail": sql_query,
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 5: UNSTRUCTURED RETRIEVER
# ═══════════════════════════════════════════════════════════════════════════════


def unstructured_retriever(state: GraphState) -> dict[str, Any]:
    """
    Semantic search over Pinecone vector store.

    Searches within the client's namespace for relevant narrative chunks.
    """
    start = time.time()
    query = state["reformulated_query"]
    client_id = state["extracted_clients"][0]

    results = retrieve_unstructured(query, client_id)
    duration = (time.time() - start) * 1000

    sources = [s.model_dump() for s in build_sources_from_vector(results)]

    return {
        "sql_query": None,
        "sql_results": [],
        "sql_error": None,
        "vector_results": results,
        "formatted_sql": "No structured data retrieved (unstructured-only route).",
        "formatted_vector": format_vector_results_for_synthesis(results),
        "sources": sources,
        "retrieval_trace": state.get("retrieval_trace", []) + [
            {
                "node": "unstructured_retriever",
                "action": f"Vector search in '{client_id}' → {len(results)} results",
                "duration_ms": duration,
                "result_count": len(results),
                "detail": f"Top score: {results[0]['score']:.3f}" if results else "No results",
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 6: HYBRID RETRIEVER
# ═══════════════════════════════════════════════════════════════════════════════


def hybrid_retriever(state: GraphState) -> dict[str, Any]:
    """
    Combined structured + unstructured retrieval.

    For cross-client queries: executes per-client isolation.
    For single-client hybrid: queries both stores for completeness.
    """
    start = time.time()
    query = state["reformulated_query"]
    clients = state["extracted_clients"]
    strategy = state["route"]
    sub_queries = state.get("sub_queries", [])

    result = retrieve_hybrid(
        query=query,
        clients=clients,
        strategy=strategy,
        sub_queries=sub_queries if sub_queries else None,
        needs_narrative=True,
    )

    duration = (time.time() - start) * 1000
    sources = [s.model_dump() for s in result["sources"]]

    return {
        "sql_query": result["sql_query"],
        "sql_results": result["sql_results"],
        "sql_error": result["sql_error"],
        "vector_results": result["vector_results"],
        "formatted_sql": result["formatted_sql"],
        "formatted_vector": result["formatted_vector"],
        "sources": sources,
        "retrieval_trace": state.get("retrieval_trace", []) + [
            {
                "node": "hybrid_retriever",
                "action": (
                    f"Hybrid retrieval across {len(clients)} client(s) → "
                    f"{len(result['sql_results'])} SQL rows, "
                    f"{len(result['vector_results'])} vector results"
                ),
                "duration_ms": duration,
                "result_count": len(result["sql_results"]) + len(result["vector_results"]),
                "detail": result.get("sql_query"),
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 7: ANSWER SYNTHESIZER
# ═══════════════════════════════════════════════════════════════════════════════


def answer_synthesizer(state: GraphState) -> dict[str, Any]:
    """
    Generate the final grounded answer using gpt-4.1.

    Receives all retrieved context (SQL + vector) and produces an answer
    that is strictly grounded in the provided data with citations.
    """
    start = time.time()
    query = state["reformulated_query"]
    route_decision = state.get("route_decision", {})

    # Build synthesis context
    system_prompt = SYNTHESIS_SYSTEM.format(
        intent=route_decision.get("intent", "unknown"),
        strategy=state.get("route", "unknown"),
        route_reasoning=state.get("route_reasoning", ""),
        clients=", ".join(state.get("extracted_clients", ["unknown"])),
    )

    # Add retry context if this is a retry attempt
    retry_count = state.get("retry_count", 0)
    if retry_count > 0 and state.get("error"):
        system_prompt += SYNTHESIS_RETRY_SUFFIX.format(
            issues=state["error"],
            allowed_clients=", ".join(state.get("extracted_clients", [])),
        )

    user_prompt = SYNTHESIS_USER.format(
        sql_results=state.get("formatted_sql", "No structured data."),
        vector_results=state.get("formatted_vector", "No unstructured data."),
        query=query,
    )

    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.synthesis_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Answer synthesis failed: {e}")
        answer = (
            "I apologize, but I encountered an error generating the answer. "
            "Please try rephrasing your question."
        )

    duration = (time.time() - start) * 1000
    logger.info(f"Answer synthesized ({duration:.0f}ms, {len(answer)} chars)")

    return {
        "answer": answer,
        "retrieval_trace": state.get("retrieval_trace", []) + [
            {
                "node": "answer_synthesizer",
                "action": f"Generated answer ({len(answer)} chars)",
                "duration_ms": duration,
                "result_count": 0,
                "detail": f"Model: {settings.synthesis_model}",
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 8: GUARDRAILS CHECKER
# ═══════════════════════════════════════════════════════════════════════════════


def guardrails_checker(state: GraphState) -> dict[str, Any]:
    """
    Post-generation quality validation (rule-based, no LLM cost).

    Checks:
    1. Answer is non-empty and reasonable length
    2. Client isolation not violated (single-client queries)
    3. Sources/citations present
    4. No confident answer without retrieved data (hallucination signal)

    On failure: sets error and needs_fallback for retry loop.
    """
    start = time.time()
    answer = state.get("answer", "")
    clients = state.get("extracted_clients", [])
    issues: list[str] = []

    # Check 1: Non-empty answer
    if not answer or len(answer.strip()) < 20:
        issues.append("Answer too short or empty")

    # Check 2: Client isolation (single-client queries only)
    if len(clients) == 1 and clients[0] != "unknown":
        other_client = "horizon" if clients[0] == "aurora" else "aurora"
        # Check for other client name in answer (but not in source citations)
        answer_without_sources = answer.split("Sources:")[0] if "Sources:" in answer else answer
        if other_client in answer_without_sources.lower():
            issues.append(
                f"Client isolation concern: answer mentions '{other_client}' "
                f"for a single-client query (client: {clients[0]})"
            )

    # Check 3: Citation presence
    has_citation = (
        "[1]" in answer
        or "source" in answer.lower()
        or "Sources:" in answer
    )
    if not has_citation and len(answer) > 50:
        issues.append("Answer lacks source citations")

    # Check 4: Hallucination detection — confident answer with no data
    has_data = bool(state.get("sql_results")) or bool(state.get("vector_results"))
    if not has_data:
        hedging_phrases = ["don't have", "no data", "not available", "cannot find", "no information"]
        has_hedging = any(phrase in answer.lower() for phrase in hedging_phrases)
        if not has_hedging:
            issues.append("No data retrieved but answer appears confident (possible hallucination)")

    duration = (time.time() - start) * 1000
    retry_count = state.get("retry_count", 0)

    if issues and retry_count < settings.max_retries:
        logger.warning(f"Guardrails failed: {issues}. Triggering retry {retry_count + 1}.")
        return {
            "error": "; ".join(issues),
            "needs_fallback": True,
            "retry_count": retry_count + 1,
            "retrieval_trace": state.get("retrieval_trace", []) + [
                {
                    "node": "guardrails_checker",
                    "action": f"FAILED: {'; '.join(issues)}",
                    "duration_ms": duration,
                    "result_count": 0,
                    "detail": f"Retry {retry_count + 1}/{settings.max_retries}",
                }
            ],
        }

    if issues:
        logger.warning(f"Guardrails issues (max retries reached): {issues}")

    return {
        "error": None,
        "needs_fallback": False,
        "retrieval_trace": state.get("retrieval_trace", []) + [
            {
                "node": "guardrails_checker",
                "action": "All checks passed" if not issues else f"Issues (max retries): {'; '.join(issues)}",
                "duration_ms": duration,
                "result_count": 0,
                "detail": None,
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 9: CONVERSATIONAL RESPONDER
# ═══════════════════════════════════════════════════════════════════════════════

_CONVERSATIONAL_SYSTEM = (
    "You are a friendly Document Retrieval QA assistant that helps users "
    "query product specifications for paint industry clients (Aurora Paints, Horizon Coatings). "
    "You can answer questions about VOC limits, lead content, drying times, regional regulations, and more.\n\n"
    "Respond naturally to the user's message. Keep it brief (2-4 sentences). "
    "If it's a greeting or introduction, suggest 1-2 example questions they could ask. "
    "Use markdown formatting.\n\n"
    "IMPORTANT: Always respond in the same language the user writes in. "
    "If they write in Dutch, reply in Dutch. If German, reply in German. And so on.\n\n"
    "CONVERSATION HISTORY:\n{chat_history}"
)


def conversational_responder(state: GraphState) -> dict[str, Any]:
    """
    Handle conversational messages (greetings, thanks, chitchat) without retrieval.

    Uses gpt-4o-mini with chat history for context-aware responses.
    """
    start = time.time()
    query = state.get("reformulated_query", state["query"])
    chat_history = _format_chat_history(state.get("chat_history", []))

    try:
        client = openai.OpenAI(api_key=settings.openai_api_key, timeout=15.0)
        response = client.chat.completions.create(
            model=settings.routing_model,
            messages=[
                {"role": "system", "content": _CONVERSATIONAL_SYSTEM.format(chat_history=chat_history)},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=250,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Conversational responder failed: {e}")
        answer = "Hello! I'm your Document Retrieval QA assistant. Ask me anything about product specifications."

    duration = (time.time() - start) * 1000

    return {
        "answer": answer,
        "sql_results": [],
        "vector_results": [],
        "formatted_sql": "",
        "formatted_vector": "",
        "sources": [],
        "retrieval_trace": state.get("retrieval_trace", []) + [
            {
                "node": "conversational_responder",
                "action": f"Conversational response ({duration:.0f}ms)",
                "duration_ms": duration,
                "result_count": 0,
                "detail": None,
            }
        ],
    }


def stream_conversational(state: dict):
    """
    Stream a conversational response token-by-token (for the streaming UI path).

    Used when the retrieval-only workflow routes to conversational —
    synthesis is handled here instead of stream_synthesis.
    """
    query = state.get("reformulated_query", state.get("query", ""))
    chat_history = _format_chat_history(state.get("chat_history", []))

    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)
        stream = client.chat.completions.create(
            model=settings.routing_model,
            messages=[
                {"role": "system", "content": _CONVERSATIONAL_SYSTEM.format(chat_history=chat_history)},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=250,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        logger.error(f"Streaming conversational failed: {e}")
        yield "Hello! I'm your Document Retrieval QA assistant. Ask me anything about product specifications."


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def stream_synthesis(state: dict):
    """
    Stream the answer synthesis token-by-token using OpenAI streaming API.

    This is called separately from the workflow — after retrieval is complete.
    Yields string chunks as they arrive from the LLM.

    Args:
        state: The workflow state dict after retrieval (contains formatted_sql,
               formatted_vector, route_decision, extracted_clients, etc.)

    Yields:
        str: Token chunks from the LLM response
    """
    query = state.get("reformulated_query", state.get("query", ""))
    route_decision = state.get("route_decision", {})

    system_prompt = SYNTHESIS_SYSTEM.format(
        intent=route_decision.get("intent", "unknown"),
        strategy=state.get("route", "unknown"),
        route_reasoning=state.get("route_reasoning", ""),
        clients=", ".join(state.get("extracted_clients", ["unknown"])),
    )

    user_prompt = SYNTHESIS_USER.format(
        sql_results=state.get("formatted_sql", "No structured data."),
        vector_results=state.get("formatted_vector", "No unstructured data."),
        query=query,
    )

    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)
        stream = client.chat.completions.create(
            model=settings.synthesis_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        logger.error(f"Streaming synthesis failed: {e}")
        yield f"I apologize, but I encountered an error generating the answer: {e}"


def _format_chat_history(messages: list) -> str:
    """Format LangGraph message list into readable text for prompts."""
    if not messages:
        return "(no previous conversation)"

    lines = []
    for msg in messages[-6:]:  # Last 3 turns max
        if hasattr(msg, "type"):
            role = "User" if msg.type == "human" else "Assistant"
            content = msg.content if hasattr(msg, "content") else str(msg)
        elif isinstance(msg, dict):
            role = "User" if msg.get("type") == "human" else "Assistant"
            content = msg.get("content", "")
        else:
            continue
        lines.append(f"{role}: {content[:200]}")

    return "\n".join(lines)
