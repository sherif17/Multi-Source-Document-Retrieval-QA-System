"""
XLSX ingestion with automatic schema detection and harmonization.

Handles the core challenge: Aurora and Horizon use different column names
for logically identical data. This module detects which schema variant
a file uses and normalizes it to the canonical ProductSpec model.

Data flow:
  XLSX file → pandas DataFrame → schema detection → column rename → 
  normalization → Pydantic validation → list[ProductSpec]
"""

from pathlib import Path

import pandas as pd

from src.models import ProductSpec, SourceType
from src.utils.logger import logger

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Maps source columns → canonical column names
SCHEMA_VARIANTS: dict[str, dict[str, str]] = {
    "aurora": {
        "client_name": "client_id",
        "product_name": "product_name",
        "region": "region",
        "parameter": "parameter",
        "value": "value",
        "unit": "unit",
        "limit_type": "limit_type",
        "notes": "notes",
    },
    "horizon": {
        "supplier_name": "client_id",
        "product_line": "product_name",
        "market": "region",
        "metric": "parameter",
        "metric_value": "value",
        "metric_unit": "unit",
        "classification": "limit_type",
        "remarks": "notes",
    },
}

# Fingerprints: unique columns that identify each schema variant
SCHEMA_FINGERPRINTS: dict[str, set[str]] = {
    "aurora": {"client_name", "product_name", "limit_type"},
    "horizon": {"supplier_name", "product_line", "metric_value"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


def detect_schema_variant(columns: list[str]) -> str:
    """
    Auto-detect which schema variant a DataFrame uses.

    Matches column names against known fingerprints. Designed to be
    extensible — adding a new client requires only a new entry in
    SCHEMA_VARIANTS and SCHEMA_FINGERPRINTS.

    Raises ValueError if no known schema matches.
    """
    col_set = {c.lower().strip() for c in columns}

    for variant, fingerprint in SCHEMA_FINGERPRINTS.items():
        if fingerprint.issubset(col_set):
            return variant

    raise ValueError(
        f"Cannot detect schema variant from columns: {columns}. "
        f"Known variants: {list(SCHEMA_FINGERPRINTS.keys())}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_xlsx(file_path: Path) -> list[ProductSpec]:
    """
    Load an XLSX file, detect its schema, normalize, and return ProductSpec records.

    Steps:
    1. Read with pandas (openpyxl engine)
    2. Detect schema variant from column names
    3. Rename columns to canonical names
    4. Validate each row through Pydantic ProductSpec model
    5. Skip rows that fail validation (log warnings)

    Returns validated ProductSpec instances ready for DB insertion.
    """
    logger.info(f"Loading XLSX: {file_path.name}")

    # Read the spreadsheet
    df = pd.read_excel(file_path, engine="openpyxl")
    logger.debug(f"  Raw shape: {df.shape[0]} rows × {df.shape[1]} cols")

    # Normalize column names (lowercase, strip whitespace)
    df.columns = [c.lower().strip() for c in df.columns]

    # Detect schema
    variant = detect_schema_variant(df.columns.tolist())
    logger.info(f"  Detected schema variant: '{variant}'")

    # Rename to canonical columns
    column_mapping = SCHEMA_VARIANTS[variant]
    df = df.rename(columns=column_mapping)

    # Add provenance metadata
    source_file = file_path.name

    # Validate each row through Pydantic
    specs: list[ProductSpec] = []
    warnings: list[str] = []

    for idx, row in df.iterrows():
        try:
            spec = ProductSpec(
                client_id=row.get("client_id", ""),
                product_name=str(row.get("product_name", "")).strip(),
                region=str(row.get("region", "")).strip(),
                parameter=str(row.get("parameter", "")).strip(),
                value=row.get("value") if pd.notna(row.get("value")) else None,
                unit=str(row.get("unit", "")).strip() if pd.notna(row.get("unit")) else None,
                limit_type=str(row.get("limit_type", "")).strip() if pd.notna(row.get("limit_type")) else None,
                notes=str(row.get("notes", "")).strip() if pd.notna(row.get("notes")) else None,
                source_file=source_file,
                source_type=SourceType.XLSX,
            )
            specs.append(spec)
        except Exception as e:
            warnings.append(f"Row {idx}: validation failed - {e}")

    if warnings:
        for w in warnings[:5]:
            logger.warning(f"  {w}")
        if len(warnings) > 5:
            logger.warning(f"  ... and {len(warnings) - 5} more warnings")

    logger.info(f"  Successfully validated {len(specs)}/{df.shape[0]} rows")
    return specs
