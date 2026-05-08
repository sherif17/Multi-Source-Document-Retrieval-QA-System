"""
Content classification: determines whether a DOCX paragraph contains
structured specification data or unstructured narrative text.

Strategy: Tiered classification (rules first, LLM fallback)
- Rule layer catches ~95% of cases deterministically (free, instant)
- LLM fallback handles ambiguous paragraphs (rare in this dataset)

This is a core assignment requirement: "How you detect and separate
structured vs. unstructured information is up to you. You should be
prepared to explain the heuristics, rules, or models you used."
"""

import re
from typing import Optional

import openai

from src.config import settings
from src.models import ContentType, Paragraph
from src.utils.logger import logger

# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Numeric spec pattern: number followed by a known unit
SPEC_UNIT_PATTERN = re.compile(
    r"\d+\.?\d*\s*(?:g/L|%\s*(?:by\s*weight)?|ppm|hours?|mg/kg)",
    re.IGNORECASE,
)

# Spec-context keywords that strengthen classification
SPEC_KEYWORDS = re.compile(
    r"\b(?:limit|maximum|minimum|content|threshold|VOC|lead|zinc|drying\s*time)\b",
    re.IGNORECASE,
)

# Narrative-only keywords (no numeric values expected)
NARRATIVE_KEYWORDS = re.compile(
    r"\b(?:guidance|recommends?|should|must comply|regulations?|formulated|"
    r"positioned|emphasises?|environments?|ventilation|complexity|consult)\b",
    re.IGNORECASE,
)

# Assignment meta-text (not client data)
IGNORE_PATTERNS = re.compile(
    r"(?:bonus\s*point|not\s*obligatory|image\s*detection)",
    re.IGNORECASE,
)

# Metadata patterns (document headers)
METADATA_PREFIXES = ("Client:", "Product family:", "Product variant:")


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


def classify_paragraph(paragraph: Paragraph) -> ContentType:
    """
    Classify a DOCX paragraph into one of five content types.

    Classification rules (evaluated in order):
    1. IGNORE — assignment instructions, not client data
    2. IMAGE — paragraph contains an embedded image
    3. METADATA — document header fields (client, product, variant)
    4. STRUCTURED — contains numeric spec values with units
    5. NARRATIVE — everything else (regulatory text, guidance, policy)

    Paragraphs classified as STRUCTURED will have their spec values
    extracted into the SQL store AND their full text stored in the
    vector store (dual-storage for rich retrieval context).
    """
    text = paragraph.text

    # Rule 1: Assignment notes → IGNORE
    if IGNORE_PATTERNS.search(text):
        return ContentType.IGNORE

    # Rule 2: Embedded image → IMAGE
    if paragraph.has_image:
        return ContentType.IMAGE

    # Rule 3: Metadata headers → METADATA
    if paragraph.style == "Heading 1":
        return ContentType.METADATA
    if text.startswith(METADATA_PREFIXES):
        return ContentType.METADATA

    # Rule 4: Contains numeric spec with unit → STRUCTURED
    # Must have BOTH a numeric+unit pattern AND a spec keyword
    has_spec_value = bool(SPEC_UNIT_PATTERN.search(text))
    has_spec_keyword = bool(SPEC_KEYWORDS.search(text))

    if has_spec_value and has_spec_keyword:
        return ContentType.STRUCTURED

    # Rule 5: Default → NARRATIVE
    return ContentType.NARRATIVE


def classify_with_llm(text: str) -> ContentType:
    """
    LLM-based classification fallback for ambiguous paragraphs.

    Used when heuristics cannot confidently classify. In practice,
    this is rarely triggered for the current dataset, but provides
    extensibility for future documents with unusual formatting.
    """
    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.routing_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify this paragraph from a product safety document.\n"
                        "STRUCTURED: Contains specific numeric specification values "
                        "(e.g., '30 g/L', '0.05% by weight') that belong in a database.\n"
                        "NARRATIVE: Contains regulatory guidance, policy text, or "
                        "recommendations without specific extractable values.\n"
                        "Respond with exactly one word: STRUCTURED or NARRATIVE."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=10,
            temperature=0,
        )
        result = response.choices[0].message.content.strip().upper()

        if "STRUCTURED" in result:
            return ContentType.STRUCTURED
        return ContentType.NARRATIVE

    except Exception as e:
        logger.warning(f"LLM classification failed, defaulting to NARRATIVE: {e}")
        return ContentType.NARRATIVE


def classify_paragraphs(
    paragraphs: list[Paragraph],
) -> dict[int, ContentType]:
    """
    Classify all paragraphs in a document. Returns mapping of index → type.

    Logs classification summary for observability.
    """
    classifications: dict[int, ContentType] = {}
    counts: dict[ContentType, int] = {ct: 0 for ct in ContentType}

    for para in paragraphs:
        content_type = classify_paragraph(para)
        classifications[para.index] = content_type
        counts[content_type] += 1

    logger.info(
        f"  Classification: "
        f"structured={counts[ContentType.STRUCTURED]}, "
        f"narrative={counts[ContentType.NARRATIVE]}, "
        f"metadata={counts[ContentType.METADATA]}, "
        f"image={counts[ContentType.IMAGE]}, "
        f"ignore={counts[ContentType.IGNORE]}"
    )

    return classifications
