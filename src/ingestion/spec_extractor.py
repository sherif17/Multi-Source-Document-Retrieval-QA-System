"""
Specification extraction from DOCX narrative text.

Extracts structured spec values (product, region, parameter, value, unit)
from paragraphs that contain inline specifications. Uses a tiered approach:

1. Regex extraction (fast, deterministic, handles known patterns)
2. LLM structured extraction (fallback for complex/novel sentences)

Example input:
  "For EcoSafe Kitchen & Bath in the EU, Aurora Paints sets an internal
   VOC limit of 32 g/L for typical residential projects"

Expected output:
  ProductSpec(product_name="EcoSafe Kitchen & Bath", region="EU",
             parameter="max_voc_content", value=32.0, unit="g/L")

Tradeoff: Regex catches ~90% of patterns in this dataset (free, instant).
LLM handles the remaining edge cases (~$0.001 per call).
"""

import json
import re
from typing import Optional

import openai

from src.config import settings
from src.models import ExtractedSpec, ProductSpec, SourceType
from src.utils.logger import logger

# ═══════════════════════════════════════════════════════════════════════════════
# REGEX PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Pattern: "VOC content/limit/threshold of X g/L"
VOC_PATTERN = re.compile(
    r"(?:VOC|voc)\s+(?:content|limit|threshold)\s+of\s+"
    r"(\d+\.?\d*)\s*(g/L)",
    re.IGNORECASE,
)

# Pattern: "lead content of X % by weight"
LEAD_PATTERN = re.compile(
    r"lead\s+content\s+of\s+(\d+\.?\d*)\s*(%\s*(?:by\s*weight)?)",
    re.IGNORECASE,
)

# Pattern: "zinc content of X % by weight"
ZINC_PATTERN = re.compile(
    r"zinc\s+content\s+of\s+(\d+\.?\d*)\s*(%\s*(?:by\s*weight)?)",
    re.IGNORECASE,
)

# Pattern: "drying time/period ... X hours"
DRYING_PATTERN = re.compile(
    r"(?:drying\s+time|airing.out\s+period)[^.]*?(\d+\.?\d*)\s*(hours?)",
    re.IGNORECASE,
)

# Product name extraction: "For {PRODUCT} in the {REGION}"
PRODUCT_REGION_PATTERN = re.compile(
    r"[Ff]or\s+(.+?)\s+(?:sold\s+)?(?:in\s+the\s+)(EU|US|GCC|European Union)",
    re.IGNORECASE,
)

# Alternative: "{PRODUCT} in the {REGION}, {CLIENT} ..."
PRODUCT_REGION_ALT = re.compile(
    r"[Ff]or\s+(.+?)\s+in\s+the\s+(EU|US|GCC)",
    re.IGNORECASE,
)

# Parameter normalization mapping
PARAMETER_MAP = {
    "voc": "max_voc_content",
    "lead": "max_lead_content",
    "zinc": "max_zinc_content",
    "drying": "recommended_drying_time_before_occupancy",
}


# ═══════════════════════════════════════════════════════════════════════════════
# REGEX EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════


def extract_specs_regex(text: str) -> list[ExtractedSpec]:
    """
    Extract spec values from text using regex patterns.

    Handles the common patterns observed in the assignment DOCX files:
    - "VOC limit of 30 g/L"
    - "lead content of 0.05 % by weight"
    - "zinc content of 0.02 % by weight"
    - "airing-out period ... 24 hours"

    Returns list of ExtractedSpec (may be empty if no patterns match).
    """
    specs: list[ExtractedSpec] = []

    # Extract product and region context
    product, region = _extract_context(text)

    # Try each parameter pattern
    patterns = [
        (VOC_PATTERN, "voc"),
        (LEAD_PATTERN, "lead"),
        (ZINC_PATTERN, "zinc"),
        (DRYING_PATTERN, "drying"),
    ]

    for pattern, param_key in patterns:
        match = pattern.search(text)
        if match:
            value_str = match.group(1)
            unit_raw = match.group(2)

            # Normalize unit
            unit = _normalize_unit(unit_raw)

            specs.append(
                ExtractedSpec(
                    product_name=product,
                    region=region,
                    parameter=PARAMETER_MAP[param_key],
                    value=float(value_str),
                    unit=unit,
                    confidence="high",
                )
            )

    return specs


def _extract_context(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract product name and region from surrounding text context."""
    product = None
    region = None

    match = PRODUCT_REGION_PATTERN.search(text)
    if not match:
        match = PRODUCT_REGION_ALT.search(text)

    if match:
        product = match.group(1).strip()
        region_raw = match.group(2).strip()
        region = _normalize_region(region_raw)

        # Clean product name (remove trailing commas, articles)
        product = re.sub(r"[,;]$", "", product).strip()
        product = re.sub(r"^(?:the|an?)\s+", "", product, flags=re.IGNORECASE).strip()

    return product, region


def _normalize_unit(unit_raw: str) -> str:
    """Normalize unit strings to canonical form."""
    unit = unit_raw.strip()
    if unit.lower() in ("g/l", "g/L"):
        return "g/L"
    if "%" in unit:
        return "% by weight"
    if "hour" in unit.lower():
        return "hours"
    return unit


def _normalize_region(region_raw: str) -> str:
    """Normalize region strings."""
    mapping = {
        "eu": "EU",
        "european union": "EU",
        "us": "US",
        "usa": "US",
        "gcc": "GCC",
    }
    return mapping.get(region_raw.lower(), region_raw.upper())


# ═══════════════════════════════════════════════════════════════════════════════
# LLM EXTRACTION (FALLBACK)
# ═══════════════════════════════════════════════════════════════════════════════


def extract_specs_llm(text: str) -> list[ExtractedSpec]:
    """
    LLM-based structured extraction for text that regex cannot handle.

    Uses gpt-4o-mini with JSON mode for cost efficiency.
    Only called when regex finds a numeric pattern but can't structure it.
    """
    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.routing_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract product specification values from this text.\n"
                        "Return a JSON object with a 'specs' array. Each spec has:\n"
                        "- product_name: exact product name mentioned\n"
                        "- region: EU, US, or GCC\n"
                        "- parameter: one of [max_voc_content, max_lead_content, "
                        "max_zinc_content, recommended_drying_time_before_occupancy]\n"
                        "- value: numeric value (float)\n"
                        "- unit: g/L, % by weight, or hours\n\n"
                        "If no spec values found, return {\"specs\": []}"
                    ),
                },
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=500,
        )

        result = json.loads(response.choices[0].message.content)
        specs_data = result.get("specs", [])

        return [
            ExtractedSpec(
                product_name=s.get("product_name"),
                region=s.get("region"),
                parameter=s.get("parameter"),
                value=float(s["value"]) if s.get("value") is not None else None,
                unit=s.get("unit"),
                confidence="medium",
            )
            for s in specs_data
            if s.get("value") is not None
        ]

    except Exception as e:
        logger.warning(f"LLM spec extraction failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def extract_specs_from_text(
    text: str,
    client_id: str,
    source_file: str,
    default_product: Optional[str] = None,
    default_region: Optional[str] = None,
) -> list[ProductSpec]:
    """
    Extract specification values from a text paragraph.

    Tiered approach:
    1. Try regex extraction (fast, free, deterministic)
    2. If regex finds nothing but text has numeric patterns, try LLM

    Enriches results with defaults from document metadata if extraction
    couldn't determine product/region from context.

    Returns fully validated ProductSpec instances ready for DB insertion.
    """
    # Step 1: Regex extraction
    extracted = extract_specs_regex(text)

    # Step 2: LLM fallback if regex found nothing but numbers exist
    if not extracted:
        has_numeric = bool(re.search(r"\d+\.?\d*\s*(?:g/L|%|hours)", text, re.IGNORECASE))
        has_spec_keyword = bool(re.search(r"\b(?:limit|content|threshold)\b", text, re.IGNORECASE))
        if has_numeric and has_spec_keyword:
            logger.debug("  Regex found nothing, trying LLM extraction")
            extracted = extract_specs_llm(text)

    # Step 3: Enrich with defaults and validate
    product_specs: list[ProductSpec] = []
    for spec in extracted:
        try:
            product_specs.append(
                ProductSpec(
                    client_id=client_id,
                    product_name=spec.product_name or default_product or "Unknown",
                    region=spec.region or default_region or "Unknown",
                    parameter=spec.parameter or "unknown",
                    value=spec.value,
                    unit=spec.unit,
                    limit_type="internal_limit",
                    notes=f"Extracted from narrative: '{text[:100]}...'",
                    source_file=source_file,
                    source_type=SourceType.DOCX_TEXT,
                )
            )
        except Exception as e:
            logger.warning(f"  Spec validation failed: {e}")

    if product_specs:
        logger.debug(f"  Extracted {len(product_specs)} specs from text")

    return product_specs
