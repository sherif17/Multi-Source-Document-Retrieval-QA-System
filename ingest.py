"""
CLI entry point for the ingestion pipeline.

Usage:
    python ingest.py              # Ingest all files from docs/
    python ingest.py --docs-dir /path/to/docs  # Custom directory

Runs the complete pipeline:
  XLSX → schema harmonization → Neon DB
  DOCX → classification → spec extraction → Neon DB + Pinecone
  Images → OCR/Vision → Neon DB + Pinecone
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.pipeline import run_ingestion_pipeline
from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Run the document ingestion pipeline"
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=None,
        help="Directory containing XLSX and DOCX files (default: ./docs)",
    )
    args = parser.parse_args()

    try:
        report = run_ingestion_pipeline(docs_dir=args.docs_dir)

        if report.errors:
            logger.error(f"Pipeline completed with {len(report.errors)} errors")
            for err in report.errors:
                logger.error(f"  • {err}")
            sys.exit(1)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
