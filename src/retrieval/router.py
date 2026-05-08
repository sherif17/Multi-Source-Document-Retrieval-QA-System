"""
Query router: determines which retrieval strategy to use for each query.

Two-tier architecture:
1. Pre-router (rule-based): Fast deterministic signal extraction
   - Client detection, cross-client detection, keyword signals
   - Free, instant, 100% reliable for what it handles

2. LLM router (gpt-4o-mini): Deep intent classification
   - Structured output for type-safe route decisions
   - Handles ambiguous queries that rules can't classify
   - Returns confidence score for fallback logic

Confidence-based behavior:
  >= 0.8: Execute chosen route directly
  0.5-0.8: Execute chosen route + secondary store as backup
  < 0.5: Default to hybrid (query both stores)
"""

import json
import re

import openai

from src.config import settings
from src.models import PreRouteSignals, RouteDecision
from src.utils.logger import logger
from src.prompts import ROUTER_SYSTEM, ROUTER_USER


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-ROUTER (RULE-BASED)
# ═══════════════════════════════════════════════════════════════════════════════


def pre_route(query: str) -> PreRouteSignals:
    """
    Extract deterministic signals from the query using pattern matching.

    These signals inform the LLM router (passed as context) and provide
    fast, free classification for clear-cut cases.
    """
    query_lower = query.lower()

    # Client extraction
    clients: list[str] = []
    if re.search(r"\baurora\b", query_lower):
        clients.append("aurora")
    if re.search(r"\bhorizon\b", query_lower):
        clients.append("horizon")

    # Cross-client detection
    is_cross_client = len(clients) > 1 or bool(
        re.search(
            r"\b(comparing|which client|stricter|vs\b|between.*and|across.*clients)\b",
            query_lower,
        )
    )

    # Numeric/structured signals
    has_numeric_ask = bool(
        re.search(
            r"\b(maximum|minimum|limit|content|value|how much|what is the|VOC|zinc|lead|"
            r"drying time|threshold)\b",
            query_lower,
        )
    )

    # Aggregation signals
    has_aggregation = bool(
        re.search(
            r"\b(across|maximum.*across|compare|which.*stricter|between|highest|lowest)\b",
            query_lower,
        )
    )

    # Narrative/unstructured signals
    has_narrative_ask = bool(
        re.search(
            r"\b(guidance|recommend|summarise|summarize|according to|policy|"
            r"consideration|types of rooms|sensitive environment|usage|ventilation|"
            r"airing.out|application)\b",
            query_lower,
        )
    )

    # Product list signals
    has_product_list = bool(
        re.search(r"(considering|including).+(and|,)", query_lower)
    )

    signals = PreRouteSignals(
        extracted_clients=clients if clients else ["unknown"],
        has_numeric_ask=has_numeric_ask,
        has_aggregation=has_aggregation,
        has_narrative_ask=has_narrative_ask,
        is_cross_client=is_cross_client,
        has_product_list=has_product_list,
    )

    logger.debug(
        f"Pre-route signals: clients={signals.extracted_clients}, "
        f"numeric={signals.has_numeric_ask}, narrative={signals.has_narrative_ask}, "
        f"cross_client={signals.is_cross_client}"
    )

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# LLM ROUTER
# ═══════════════════════════════════════════════════════════════════════════════


def llm_route(query: str, signals: PreRouteSignals) -> RouteDecision:
    """
    Classify query intent and select retrieval strategy using gpt-4o-mini.

    Uses structured JSON output for type-safe parsing. The pre-route signals
    are passed as context to help the LLM make better decisions.
    """
    # Format signals for prompt
    signals_text = (
        f"Clients detected: {signals.extracted_clients}\n"
        f"Has numeric ask: {signals.has_numeric_ask}\n"
        f"Has aggregation: {signals.has_aggregation}\n"
        f"Has narrative ask: {signals.has_narrative_ask}\n"
        f"Is cross-client: {signals.is_cross_client}\n"
        f"Has product list: {signals.has_product_list}"
    )

    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.routing_model,
            messages=[
                {
                    "role": "system",
                    "content": ROUTER_SYSTEM.format(pre_route_signals=signals_text),
                },
                {
                    "role": "user",
                    "content": ROUTER_USER.format(query=query),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=500,
        )

        result = json.loads(response.choices[0].message.content)

        # Parse into typed model with validation
        decision = RouteDecision(
            intent=result.get("intent", "hybrid"),
            strategy=result.get("strategy", "hybrid_sql_primary"),
            clients=result.get("clients", signals.extracted_clients),
            reasoning=result.get("reasoning", ""),
            confidence=float(result.get("confidence", 0.5)),
            sub_queries=result.get("sub_queries", []),
            needs_aggregation=result.get("needs_aggregation", False),
            needs_narrative_context=result.get("needs_narrative_context", False),
        )

        logger.info(
            f"Route decision: intent={decision.intent}, strategy={decision.strategy}, "
            f"confidence={decision.confidence:.2f}, clients={decision.clients}"
        )
        logger.debug(f"  Reasoning: {decision.reasoning}")

        return decision

    except Exception as e:
        logger.error(f"LLM routing failed: {e}. Falling back to hybrid.")
        return _fallback_route(signals)


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK ROUTING
# ═══════════════════════════════════════════════════════════════════════════════


def _fallback_route(signals: PreRouteSignals) -> RouteDecision:
    """
    Rule-based fallback when LLM router fails.

    Applies simple heuristics from pre-route signals to make a safe decision.
    Defaults to hybrid (over-retrieval) when uncertain.
    """
    # Cross-client → cross_client_hybrid
    if signals.is_cross_client:
        strategy = "cross_client_hybrid"
        intent = "comparison"
    # Only narrative → unstructured
    elif signals.has_narrative_ask and not signals.has_numeric_ask:
        strategy = "unstructured_only"
        intent = "narrative"
    # Only numeric → structured
    elif signals.has_numeric_ask and not signals.has_narrative_ask:
        strategy = "structured_only"
        intent = "lookup" if not signals.has_aggregation else "aggregation"
    # Both or neither → hybrid
    else:
        strategy = "hybrid_sql_primary"
        intent = "hybrid"

    return RouteDecision(
        intent=intent,
        strategy=strategy,
        clients=signals.extracted_clients,
        reasoning="Fallback routing based on keyword signals (LLM router unavailable)",
        confidence=0.5,
        sub_queries=[],
        needs_aggregation=signals.has_aggregation,
        needs_narrative_context=signals.has_narrative_ask,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def route_query(query: str) -> tuple[PreRouteSignals, RouteDecision]:
    """
    Full routing pipeline: pre-route signals → LLM classification.

    Returns both the signals (for trace/debug) and the route decision.
    """
    signals = pre_route(query)
    decision = llm_route(query, signals)

    # Override clients from pre-route if LLM missed them
    if decision.clients == ["unknown"] and signals.extracted_clients != ["unknown"]:
        decision.clients = signals.extracted_clients

    return signals, decision
