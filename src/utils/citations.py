"""
Citation and source formatting utilities.

Handles the transformation of raw retrieval results into
human-readable citation formats for the Streamlit UI.
"""

from src.models import Source


def format_sources_for_display(sources: list[dict]) -> str:
    """
    Format source citations for display below the answer.

    Groups by store type and provides structured references.
    """
    if not sources:
        return ""

    lines = ["**Sources:**"]
    for i, source in enumerate(sources, 1):
        store = source.get("store", "?")
        file = source.get("file", "unknown")
        detail = source.get("detail", "")
        score = source.get("relevance_score")

        tag = {"sql": "[SQL]", "vector": "[VEC]", "image_ocr": "[IMG]"}.get(store, "[?]")
        score_text = f" (relevance: {score:.2f})" if score else ""

        lines.append(f"  [{i}] {tag} {file} — {detail}{score_text}")

    return "\n".join(lines)


def format_trace_for_display(trace: list[dict]) -> list[dict]:
    """
    Format the retrieval trace for Streamlit display.

    Returns structured data ready for rendering with icons and timing.
    """
    formatted = []
    for step in trace:
        formatted.append({
            "icon": "",
            "node": step["node"],
            "action": step["action"],
            "duration_ms": step["duration_ms"],
            "result_count": step.get("result_count", 0),
            "detail": step.get("detail"),
        })

    return formatted
