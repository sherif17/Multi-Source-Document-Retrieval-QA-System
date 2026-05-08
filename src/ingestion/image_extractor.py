"""
Image extraction pipeline for embedded DOCX images using GPT-4o Vision.

Strategy: GPT-4o Vision as the sole extraction engine.
- Sends images to GPT-4o which returns structured JSON directly
- No Tesseract dependency required
- If Vision fails → log warning, continue pipeline (image is bonus feature)
"""

import base64
import json
from typing import Optional

import openai

from src.config import settings
from src.models import ExtractedSpec, ImageExtractionResult, ProductSpec, SourceType
from src.utils.logger import logger


# ═══════════════════════════════════════════════════════════════════════════════
# GPT-4o VISION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════


def extract_with_vision(image_bytes: bytes, client_id: str) -> list[ExtractedSpec]:
    """
    Extract structured spec data from an image using GPT-4o Vision.

    Sends the image to GPT-4o with a structured extraction prompt.
    Returns parsed spec values as ExtractedSpec instances.
    """
    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Detect content type from image header bytes
        content_type = _detect_image_type(image_bytes)

        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Extract ALL product specification values from this image.\n"
                                f"This image contains spec data for client '{client_id}'.\n\n"
                                f"Return a JSON object with a 'specs' array. Each spec has:\n"
                                f"- product_name: str (exact product name)\n"
                                f"- region: str (EU, US, or GCC)\n"
                                f"- parameter: str (max_voc_content, max_lead_content, "
                                f"max_zinc_content, or recommended_drying_time_before_occupancy)\n"
                                f"- value: float (the numeric value)\n"
                                f"- unit: str (g/L, % by weight, or hours)\n\n"
                                f"Extract EVERY row/entry visible. If the image doesn't contain "
                                f"spec data, return {{\"specs\": []}}."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=2000,
            temperature=0,
        )

        result = json.loads(response.choices[0].message.content)
        specs_data = result.get("specs", [])

        extracted = [
            ExtractedSpec(
                product_name=s.get("product_name"),
                region=s.get("region"),
                parameter=s.get("parameter"),
                value=float(s["value"]) if s.get("value") is not None else None,
                unit=s.get("unit"),
                confidence="high",
            )
            for s in specs_data
            if s.get("value") is not None
        ]

        logger.info(f"  Vision extracted {len(extracted)} specs from image")
        return extracted

    except Exception as e:
        logger.error(f"  Vision extraction failed: {e}")
        return []




# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def process_image(
    image_bytes: bytes,
    client_id: str,
    source_file: str,
    image_index: int = 0,
) -> tuple[list[ProductSpec], str]:
    """
    Image processing pipeline using GPT-4o Vision only.

    Returns:
    - list[ProductSpec]: Extracted specs ready for SQL insertion
    - str: Formatted text description for vector store embedding
    """
    logger.info(f"  Processing image {image_index} from {source_file}")

    # GPT-4o Vision (structured extraction)
    vision_specs = extract_with_vision(image_bytes, client_id)

    # All Vision specs get high confidence (GPT-4o is the authority)
    for spec in vision_specs:
        spec.confidence = "high"

    # Convert to ProductSpec instances
    product_specs: list[ProductSpec] = []
    for spec in vision_specs:
        try:
            product_specs.append(
                ProductSpec(
                    client_id=client_id,
                    product_name=spec.product_name or "Unknown",
                    region=spec.region or "Unknown",
                    parameter=spec.parameter or "unknown",
                    value=spec.value,
                    unit=spec.unit,
                    limit_type="internal_limit",
                    notes="Extracted from embedded image via GPT-4o Vision",
                    source_file=source_file,
                    source_type=SourceType.DOCX_IMAGE,
                )
            )
        except Exception as e:
            logger.warning(f"  Image spec validation failed: {e}")

    # Format specs as text for vector store embedding
    text_for_vector = _format_specs_as_text(vision_specs)

    logger.info(f"  Image pipeline: {len(product_specs)} specs extracted via Vision")
    return product_specs, text_for_vector


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def _detect_image_type(image_bytes: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if image_bytes[:4] == b"GIF8":
        return "image/gif"
    return "image/png"  # Default assumption


def _format_specs_as_text(specs: list[ExtractedSpec]) -> str:
    """Format extracted specs as readable text for vector embedding."""
    if not specs:
        return ""

    lines = ["Specification data extracted from embedded image:"]
    for spec in specs:
        parts = []
        if spec.product_name:
            parts.append(f"Product: {spec.product_name}")
        if spec.region:
            parts.append(f"Region: {spec.region}")
        if spec.parameter:
            parts.append(f"Parameter: {spec.parameter}")
        if spec.value is not None:
            parts.append(f"Value: {spec.value}")
        if spec.unit:
            parts.append(f"Unit: {spec.unit}")
        lines.append(" | ".join(parts))

    return "\n".join(lines)
