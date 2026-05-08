"""
Main ingestion pipeline orchestrator.

Coordinates the full data ingestion workflow:
1. XLSX files → schema harmonization → Neon DB
2. DOCX files → classify → extract specs → Neon DB + Pinecone
3. DOCX images → OCR/Vision → Neon DB + Pinecone

Design principles:
- Idempotent: truncate-and-reload strategy (safe for re-runs)
- Observable: detailed logging at each stage with final IngestionReport
- Fault-tolerant: individual file failures don't halt the pipeline
- Deterministic chunk IDs: enables Pinecone upsert without duplicates
"""

import re
import time
from pathlib import Path

from src.config import settings
from src.models import (
    ChunkMetadata,
    ChunkType,
    ContentType,
    DocumentChunk,
    IngestionReport,
    ProductSpec,
)
from src.stores import sql_store, vector_store
from src.utils.logger import logger

from .content_classifier import classify_paragraphs
from .docx_loader import extract_images_from_docx, parse_docx
from .image_extractor import process_image
from .spec_extractor import extract_specs_from_text
from .xlsx_loader import load_xlsx


# ═══════════════════════════════════════════════════════════════════════════════
# CLIENT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


def detect_client_from_filename(file_path: Path) -> str:
    """Infer client ID from filename pattern (aurora_*.xlsx → 'aurora')."""
    name = file_path.stem.lower()
    if "aurora" in name:
        return "aurora"
    elif "horizon" in name:
        return "horizon"
    raise ValueError(f"Cannot detect client from filename: {file_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTITY EXTRACTION FOR METADATA
# ═══════════════════════════════════════════════════════════════════════════════

# Known product names for metadata tagging
KNOWN_PRODUCTS = {
    "aurora": [
        "EcoSafe Interior Wall Paint",
        "EcoSafe Kitchen & Bath",
        "EcoSafe Ceiling Paint",
        "EcoSafe Exterior Facade",
        "EcoShield Floor Coating",
        "ProShield Primer",
    ],
    "horizon": [
        "UltraSafe Interior Wall Paint",
        "UltraSafe Kitchen & Bath",
        "UltraSafe Ceiling Paint",
        "UltraSafe Corridor Paint",
        "UltraShield Primer",
    ],
}

KNOWN_REGIONS = ["EU", "US", "GCC"]


def find_products_in_text(text: str, client_id: str) -> list[str]:
    """Find known product names mentioned in a text paragraph."""
    products = KNOWN_PRODUCTS.get(client_id, [])
    return [p for p in products if p.lower() in text.lower()]


def find_regions_in_text(text: str) -> list[str]:
    """Find known region identifiers mentioned in text."""
    found = []
    for region in KNOWN_REGIONS:
        if re.search(rf"\b{region}\b", text):
            found.append(region)
    return found


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS EVENT MODEL
# ═══════════════════════════════════════════════════════════════════════════════


class IngestionEvent:
    """A progress event yielded during ingestion for UI rendering."""

    def __init__(
        self,
        phase: str,
        status: str,
        message: str,
        file: str = "",
        detail: str = "",
        progress: float = 0.0,
    ):
        self.phase = phase      # "init", "xlsx", "docx", "images", "done"
        self.status = status    # "start", "progress", "success", "warning", "error"
        self.message = message
        self.file = file
        self.detail = detail
        self.progress = progress  # 0.0 to 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING PIPELINE (for Streamlit UI)
# ═══════════════════════════════════════════════════════════════════════════════


def run_ingestion_streaming(file_paths: list[Path]):
    """
    Generator-based ingestion that yields IngestionEvent objects.

    Accepts explicit file paths (from Streamlit upload) instead of
    scanning a directory. Yields progress events for real-time UI updates.

    Usage:
        for event in run_ingestion_streaming(uploaded_files):
            display(event)
    """
    report = IngestionReport()
    start_time = time.time()

    xlsx_files = [f for f in file_paths if f.suffix == ".xlsx"]
    docx_files = [f for f in file_paths if f.suffix == ".docx"]
    total_files = len(file_paths)

    # ── Phase: Initialize ──
    yield IngestionEvent("init", "start", "Initializing database connections...")
    try:
        sql_store.initialize_schema()
        vector_store.get_index()
        yield IngestionEvent("init", "success", "Connected to Neon DB + Pinecone")
    except Exception as e:
        yield IngestionEvent("init", "error", f"Store initialization failed: {e}")
        report.errors.append(str(e))
        return

    # ── Phase: Clear existing data ──
    yield IngestionEvent("init", "progress", "Clearing existing data for clean ingestion...")
    try:
        sql_store.truncate_specs()
        # Only delete namespaces that exist to avoid 404 errors
        try:
            stats = vector_store.get_index_stats()
            for ns in list(stats.get("namespaces", {}).keys()):
                vector_store.delete_namespace(ns)
        except Exception:
            pass  # Pinecone may not have namespaces yet
        yield IngestionEvent("init", "success", "Previous data cleared")
    except Exception as e:
        yield IngestionEvent("init", "warning", f"Cleanup warning: {e}")

    files_processed = 0

    # ── Phase: XLSX ──
    if xlsx_files:
        yield IngestionEvent("xlsx", "start", f"Processing {len(xlsx_files)} XLSX file(s)...")

    for xlsx_path in xlsx_files:
        yield IngestionEvent(
            "xlsx", "progress",
            f"Loading {xlsx_path.name}...",
            file=xlsx_path.name,
        )
        try:
            specs = load_xlsx(xlsx_path)
            inserted = sql_store.insert_specs(specs)
            report.total_xlsx_rows += inserted
            files_processed += 1

            yield IngestionEvent(
                "xlsx", "success",
                f"{xlsx_path.name}: {inserted} rows inserted into SQL",
                file=xlsx_path.name,
                detail=f"Schema detected, {inserted} specs validated & stored",
                progress=files_processed / total_files,
            )
        except Exception as e:
            report.errors.append(f"{xlsx_path.name}: {e}")
            yield IngestionEvent(
                "xlsx", "error",
                f"{xlsx_path.name}: {e}",
                file=xlsx_path.name,
            )

    # ── Phase: DOCX ──
    if docx_files:
        yield IngestionEvent("docx", "start", f"Processing {len(docx_files)} DOCX file(s)...")

    for docx_path in docx_files:
        yield IngestionEvent(
            "docx", "progress",
            f"Parsing & classifying {docx_path.name}...",
            file=docx_path.name,
        )
        try:
            _ingest_docx(docx_path, report)
            files_processed += 1

            yield IngestionEvent(
                "docx", "success",
                f"{docx_path.name}: {report.total_docx_specs_extracted} specs extracted, chunks vectorized",
                file=docx_path.name,
                detail="Paragraphs classified > structured specs to SQL, narrative to Pinecone",
                progress=files_processed / total_files,
            )
        except Exception as e:
            report.errors.append(f"{docx_path.name}: {e}")
            yield IngestionEvent(
                "docx", "error",
                f"{docx_path.name}: {e}",
                file=docx_path.name,
            )

    # ── Phase: Images ──
    has_images = any(True for dp in docx_files for _ in [1])  # Check if DOCX files exist
    if docx_files:
        yield IngestionEvent("images", "start", "Extracting embedded images from DOCX files...")

    for docx_path in docx_files:
        try:
            client_id = detect_client_from_filename(docx_path)
            images = extract_images_from_docx(docx_path)
            if not images:
                yield IngestionEvent(
                    "images", "progress",
                    f"{docx_path.name}: no embedded images found",
                    file=docx_path.name,
                )
            else:
                img_specs_before = report.total_image_specs_extracted
                _process_docx_images(docx_path, report, images=images)
                new_specs = report.total_image_specs_extracted - img_specs_before
                yield IngestionEvent(
                    "images", "success" if new_specs > 0 else "progress",
                    f"{docx_path.name}: {len(images)} image(s) processed, {new_specs} specs extracted via Vision",
                    file=docx_path.name,
                )
        except Exception as e:
            report.warnings.append(f"Image processing: {e}")
            yield IngestionEvent(
                "images", "warning",
                f"{docx_path.name} images: {e}",
                file=docx_path.name,
            )

    # ── Phase: Done ──
    report.duration_seconds = time.time() - start_time

    # Warm the distinct values cache for the dynamic SQL prompt
    try:
        sql_store.refresh_distinct_values_cache()
    except Exception:
        pass  # non-critical — cache will populate on first query

    yield IngestionEvent(
        "done", "success",
        "Ingestion complete!",
        detail=(
            f"SQL rows: {report.total_xlsx_rows + report.total_docx_specs_extracted + report.total_image_specs_extracted} | "
            f"Vector chunks: {report.total_narrative_chunks} | "
            f"Duration: {report.duration_seconds:.1f}s"
        ),
        progress=1.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH PIPELINE (for CLI use)
# ═══════════════════════════════════════════════════════════════════════════════


def run_ingestion_pipeline(docs_dir: Path | None = None) -> IngestionReport:
    """
    Execute the complete ingestion pipeline.

    Steps:
    1. Initialize stores (create tables, verify connections)
    2. Clear existing data (idempotent re-ingestion)
    3. Ingest XLSX files → Neon DB
    4. Ingest DOCX files → classify → SQL + Pinecone
    5. Extract images → OCR/Vision → SQL + Pinecone
    6. Report results

    Returns IngestionReport with counts, warnings, and timing.
    """
    start_time = time.time()
    report = IngestionReport()

    if docs_dir is None:
        docs_dir = settings.docs_dir

    logger.info("=" * 60)
    logger.info("STARTING INGESTION PIPELINE")
    logger.info(f"Source directory: {docs_dir}")
    logger.info("=" * 60)

    # ── Step 1: Initialize stores ──
    try:
        sql_store.initialize_schema()
        vector_store.get_index()  # Verify Pinecone connection
    except Exception as e:
        report.errors.append(f"Store initialization failed: {e}")
        logger.error(f"Store initialization failed: {e}")
        return report

    # ── Step 2: Clear existing data ──
    logger.info("Clearing existing data for clean re-ingestion...")
    try:
        sql_store.truncate_specs()
        vector_store.delete_namespace("aurora")
        vector_store.delete_namespace("horizon")
    except Exception as e:
        report.warnings.append(f"Data cleanup warning: {e}")
        logger.warning(f"Data cleanup warning (continuing): {e}")

    # ── Step 3: Ingest XLSX files ──
    logger.info("\n" + "─" * 40)
    logger.info("PHASE 1: XLSX INGESTION")
    logger.info("─" * 40)

    xlsx_files = sorted(docs_dir.glob("*.xlsx"))
    for xlsx_path in xlsx_files:
        try:
            specs = load_xlsx(xlsx_path)
            inserted = sql_store.insert_specs(specs)
            report.total_xlsx_rows += inserted
        except Exception as e:
            report.errors.append(f"XLSX ingestion failed for {xlsx_path.name}: {e}")
            logger.error(f"XLSX ingestion failed for {xlsx_path.name}: {e}")

    # ── Step 4: Ingest DOCX files (text) ──
    logger.info("\n" + "─" * 40)
    logger.info("PHASE 2: DOCX INGESTION")
    logger.info("─" * 40)

    docx_files = sorted(docs_dir.glob("*.docx"))
    for docx_path in docx_files:
        try:
            _ingest_docx(docx_path, report)
        except Exception as e:
            report.errors.append(f"DOCX ingestion failed for {docx_path.name}: {e}")
            logger.error(f"DOCX ingestion failed for {docx_path.name}: {e}")

    # ── Step 5: Extract and process images ──
    logger.info("\n" + "─" * 40)
    logger.info("PHASE 3: IMAGE EXTRACTION")
    logger.info("─" * 40)

    for docx_path in docx_files:
        try:
            _process_docx_images(docx_path, report)
        except Exception as e:
            report.warnings.append(f"Image processing failed for {docx_path.name}: {e}")
            logger.warning(f"Image processing failed for {docx_path.name}: {e}")

    # ── Final report ──
    report.duration_seconds = time.time() - start_time

    logger.info("\n" + "=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"  XLSX rows inserted:     {report.total_xlsx_rows}")
    logger.info(f"  DOCX specs extracted:   {report.total_docx_specs_extracted}")
    logger.info(f"  Image specs extracted:  {report.total_image_specs_extracted}")
    logger.info(f"  Narrative chunks:       {report.total_narrative_chunks}")
    logger.info(f"  Warnings:               {len(report.warnings)}")
    logger.info(f"  Errors:                 {len(report.errors)}")
    logger.info(f"  Duration:               {report.duration_seconds:.1f}s")
    logger.info("=" * 60)

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# DOCX PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════


def _ingest_docx(docx_path: Path, report: IngestionReport) -> None:
    """
    Process a single DOCX file: classify paragraphs, extract specs, create chunks.

    Dual-storage strategy:
    - STRUCTURED paragraphs → extract spec values → SQL + store full text → Pinecone
    - NARRATIVE paragraphs → store full text → Pinecone
    - METADATA/IGNORE → skip
    """
    content = parse_docx(docx_path)
    client_id = detect_client_from_filename(docx_path)
    source_file = docx_path.name

    # Classify all paragraphs
    classifications = classify_paragraphs(content.paragraphs)

    # Process each paragraph based on classification
    chunks: list[DocumentChunk] = []

    for para in content.paragraphs:
        content_type = classifications.get(para.index, ContentType.NARRATIVE)

        if content_type in (ContentType.METADATA, ContentType.IGNORE, ContentType.IMAGE):
            continue

        elif content_type == ContentType.STRUCTURED:
            # Extract spec values → SQL
            specs = extract_specs_from_text(
                text=para.text,
                client_id=client_id,
                source_file=source_file,
                default_product=content.metadata.product_family,
            )
            if specs:
                sql_store.insert_specs(specs)
                report.total_docx_specs_extracted += len(specs)

            # Also store full paragraph text in vector store (for context retrieval)
            chunks.append(
                DocumentChunk(
                    id=f"{client_id}_{docx_path.stem}_p{para.index}",
                    text=para.text,
                    metadata=ChunkMetadata(
                        client_id=client_id,
                        source_file=source_file,
                        paragraph_index=para.index,
                        chunk_type=ChunkType.SPEC_CONTEXT,
                        has_spec_value=True,
                        products_mentioned=find_products_in_text(para.text, client_id),
                        regions_mentioned=find_regions_in_text(para.text),
                    ),
                )
            )

        elif content_type == ContentType.NARRATIVE:
            chunks.append(
                DocumentChunk(
                    id=f"{client_id}_{docx_path.stem}_p{para.index}",
                    text=para.text,
                    metadata=ChunkMetadata(
                        client_id=client_id,
                        source_file=source_file,
                        paragraph_index=para.index,
                        chunk_type=ChunkType.NARRATIVE,
                        has_spec_value=False,
                        products_mentioned=find_products_in_text(para.text, client_id),
                        regions_mentioned=find_regions_in_text(para.text),
                    ),
                )
            )

    # Embed and upsert all chunks to Pinecone
    if chunks:
        upserted = vector_store.upsert_chunks(chunks, namespace=client_id)
        report.total_narrative_chunks += upserted


def _process_docx_images(
    docx_path: Path,
    report: IngestionReport,
    images: list[bytes] | None = None,
) -> None:
    """Process embedded images from a DOCX file through Vision pipeline."""
    client_id = detect_client_from_filename(docx_path)
    source_file = docx_path.name

    if images is None:
        images = extract_images_from_docx(docx_path)
    if not images:
        return

    for img_idx, image_bytes in enumerate(images):
        # Run OCR/Vision pipeline
        specs, ocr_text = process_image(
            image_bytes=image_bytes,
            client_id=client_id,
            source_file=source_file,
            image_index=img_idx,
        )

        # Insert extracted specs into SQL
        if specs:
            sql_store.insert_specs(specs)
            report.total_image_specs_extracted += len(specs)

        # Store OCR text in vector store for semantic search
        if ocr_text.strip():
            chunk = DocumentChunk(
                id=f"{client_id}_{docx_path.stem}_img{img_idx}",
                text=ocr_text,
                metadata=ChunkMetadata(
                    client_id=client_id,
                    source_file=source_file,
                    paragraph_index=-1,
                    chunk_type=ChunkType.IMAGE_OCR,
                    has_spec_value=True,
                    products_mentioned=find_products_in_text(ocr_text, client_id),
                    regions_mentioned=find_regions_in_text(ocr_text),
                ),
            )
            vector_store.upsert_chunks([chunk], namespace=client_id)
