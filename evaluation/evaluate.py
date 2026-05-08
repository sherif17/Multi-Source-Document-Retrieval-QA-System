"""
Automated evaluation script for the 5 required queries.

Runs each query through the full pipeline and checks:
1. Route correctness (did the router pick the right strategy?)
2. Answer content (does it contain expected keywords/values?)
3. Client isolation (no data leakage between clients)
4. Source attribution (are citations present?)

Usage:
    python -m evaluation.evaluate
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.workflow import get_workflow
from src.utils.logger import logger


def load_golden_answers() -> dict:
    path = Path(__file__).parent / "golden_answers.json"
    with open(path) as f:
        return json.load(f)


def evaluate_query(workflow, query_data: dict) -> dict:
    """Evaluate a single query against golden expectations."""
    query = query_data["query"]
    query_id = query_data["id"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {query_id}: {query[:80]}...")

    start = time.time()
    try:
        result = workflow.invoke({
            "query": query,
            "chat_history": [],
            "session_id": f"eval_{query_id}",
            "retrieval_trace": [],
            "retry_count": 0,
            "needs_fallback": False,
        })
        duration = time.time() - start
    except Exception as e:
        return {
            "query_id": query_id,
            "status": "ERROR",
            "error": str(e),
            "duration_s": time.time() - start,
        }

    answer = result.get("answer", "")
    route = result.get("route", "unknown")
    clients = result.get("extracted_clients", [])

    # Check 1: Route correctness
    expected_route = query_data["expected_route"]
    route_correct = route == expected_route

    # Check 2: Answer contains expected keywords
    expected_contains = query_data.get("expected_answer_contains", [])
    answer_lower = answer.lower()
    missing_keywords = [kw for kw in expected_contains if kw.lower() not in answer_lower]
    content_correct = len(missing_keywords) == 0

    # Check 3: Client isolation
    expected_client = query_data["expected_client"]
    if isinstance(expected_client, str):
        # Single-client query — check no other client mentioned in answer body
        other_client = "horizon" if expected_client == "aurora" else "aurora"
        answer_body = answer.split("Sources:")[0] if "Sources:" in answer else answer
        isolation_ok = other_client not in answer_body.lower()
    else:
        isolation_ok = True  # Cross-client query, both are expected

    # Check 4: Citation presence
    has_citations = any(marker in answer for marker in ["📊", "[1]", "Sources:"])

    eval_result = {
        "query_id": query_id,
        "status": "PASS" if (route_correct and content_correct and isolation_ok) else "PARTIAL",
        "duration_s": duration,
        "route_expected": expected_route,
        "route_actual": route,
        "route_correct": route_correct,
        "content_correct": content_correct,
        "missing_keywords": missing_keywords,
        "isolation_ok": isolation_ok,
        "has_citations": has_citations,
        "answer_length": len(answer),
        "answer_preview": answer[:200],
    }

    # Log result
    status_icon = "✅" if eval_result["status"] == "PASS" else "⚠️"
    logger.info(f"{status_icon} {query_id}: {eval_result['status']}")
    logger.info(f"   Route: {route} (expected: {expected_route}) {'✓' if route_correct else '✗'}")
    logger.info(f"   Content: {'✓' if content_correct else '✗ missing: ' + str(missing_keywords)}")
    logger.info(f"   Isolation: {'✓' if isolation_ok else '✗'}")
    logger.info(f"   Citations: {'✓' if has_citations else '✗'}")
    logger.info(f"   Duration: {duration:.1f}s")

    return eval_result


def main():
    golden = load_golden_answers()
    workflow = get_workflow()

    results = []
    for query_data in golden["queries"]:
        result = evaluate_query(workflow, query_data)
        results.append(result)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    logger.info(f"Results: {passed}/{total} PASS")

    for r in results:
        icon = "✅" if r["status"] == "PASS" else "⚠️" if r["status"] == "PARTIAL" else "❌"
        logger.info(f"  {icon} {r['query_id']}: {r['status']} ({r['duration_s']:.1f}s)")

    # Save results
    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
