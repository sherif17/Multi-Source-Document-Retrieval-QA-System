"""
DOCX parsing with paragraph extraction and metadata detection.

Extracts three types of content from Word documents:
1. Metadata paragraphs (title, client, product family)
2. Content paragraphs (text for classification downstream)
3. Image references (for OCR/Vision processing)

Design decision: This module only PARSES — it does not classify or extract specs.
Classification and extraction are separate concerns handled by content_classifier.py
and spec_extractor.py respectively. This keeps each module testable in isolation.
"""

from pathlib import Path

from docx import Document

from src.models import DocMetadata, DocxContent, Paragraph
from src.utils.logger import logger


def parse_docx(file_path: Path) -> DocxContent:
    """
    Parse a DOCX file into structured content ready for classification.

    Extracts:
    - All non-empty paragraphs with their styles and indices
    - Document-level metadata from header paragraphs
    - Image presence detection per paragraph

    Returns DocxContent with paragraphs, metadata, and image count.
    """
    logger.info(f"Parsing DOCX: {file_path.name}")

    doc = Document(str(file_path))
    paragraphs: list[Paragraph] = []
    image_count = 0

    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if not text:
            continue

        # Detect embedded images in paragraph runs
        has_image = _paragraph_has_image(para)
        if has_image:
            image_count += 1

        paragraphs.append(
            Paragraph(
                index=i,
                text=text,
                style=para.style.name,
                has_image=has_image,
            )
        )

    # Extract document metadata from header paragraphs
    metadata = _extract_metadata(paragraphs)

    logger.info(
        f"  Parsed {len(paragraphs)} paragraphs, "
        f"{image_count} images, "
        f"client='{metadata.client}'"
    )

    return DocxContent(
        file_path=str(file_path),
        paragraphs=paragraphs,
        metadata=metadata,
        image_count=image_count,
    )


def _paragraph_has_image(para) -> bool:
    """Check if a paragraph contains an embedded image (drawing element)."""
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    drawing_ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

    for run in para.runs:
        # Check for w:drawing elements (modern DOCX images)
        drawings = run._element.findall(
            f".//{{{drawing_ns.strip('{}')}}}..//"
        )
        # More reliable: check for any drawing in the run's XML
        run_xml = run._element.xml
        if "w:drawing" in run_xml or "wp:inline" in run_xml:
            return True

    return False


def _extract_metadata(paragraphs: list[Paragraph]) -> DocMetadata:
    """
    Extract document metadata from the first few paragraphs.

    Both Aurora and Horizon DOCX files follow the same header pattern:
      P0: "Global Product Safety Brief – Client {NAME} – {PRODUCT}" (Heading)
      P1: "Client: {NAME}"
      P2: "Product family: {PRODUCT}"
      P3: "Product variant: {VARIANT}"
    """
    client = "unknown"
    product_family = None
    product_variant = None

    for para in paragraphs[:6]:
        text = para.text

        if text.startswith("Client:"):
            client = text.replace("Client:", "").strip()

        elif text.startswith("Product family:"):
            product_family = text.replace("Product family:", "").strip()

        elif text.startswith("Product variant:"):
            product_variant = text.replace("Product variant:", "").strip()

        elif "Client" in text and "–" in text and para.style == "Heading 1":
            # Parse from heading: "Global Product Safety Brief – Client X – Product Y"
            parts = text.split("–")
            if len(parts) >= 2:
                for part in parts:
                    part = part.strip()
                    if part.startswith("Client"):
                        client = part.replace("Client", "").strip()

    return DocMetadata(
        client=client,
        product_family=product_family,
        product_variant=product_variant,
    )


def extract_images_from_docx(file_path: Path) -> list[bytes]:
    """
    Extract all embedded image blobs from a DOCX file.

    Returns list of raw image bytes (PNG/JPEG) for OCR/Vision processing.
    Images are extracted from the document's relationship parts.
    """
    doc = Document(str(file_path))
    images: list[bytes] = []

    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            blob = rel.target_part.blob
            images.append(blob)
            logger.debug(
                f"  Extracted image: {len(blob)} bytes, "
                f"type={rel.target_part.content_type}"
            )

    logger.info(f"  Extracted {len(images)} images from {file_path.name}")
    return images
