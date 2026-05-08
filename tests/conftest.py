"""
Shared test fixtures and configuration.

Provides:
- Sample data fixtures for unit tests
- Path helpers for test files
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

DOCS_DIR = Path(__file__).parent.parent / "docs"


@pytest.fixture
def aurora_xlsx_path():
    return DOCS_DIR / "aurora_product_data.xlsx"


@pytest.fixture
def horizon_xlsx_path():
    return DOCS_DIR / "horizon_product_data.xlsx"


@pytest.fixture
def aurora_docx_path():
    return DOCS_DIR / "aurora_product_brief.docx"


@pytest.fixture
def horizon_docx_path():
    return DOCS_DIR / "horizon_product_brief.docx"


@pytest.fixture
def sample_aurora_spec():
    """A known-good ProductSpec for Aurora."""
    from src.models import ProductSpec, SourceType

    return ProductSpec(
        client_id="aurora",
        product_name="EcoSafe Interior Wall Paint",
        region="EU",
        parameter="max_voc_content",
        value=30.0,
        unit="g/L",
        limit_type="internal_limit",
        notes="Test fixture",
        source_file="aurora_product_data.xlsx",
        source_type=SourceType.XLSX,
    )


@pytest.fixture
def sample_horizon_spec():
    """A known-good ProductSpec for Horizon."""
    from src.models import ProductSpec, SourceType

    return ProductSpec(
        client_id="horizon",
        product_name="UltraSafe Interior Wall Paint",
        region="EU",
        parameter="max_voc_content",
        value=25.0,
        unit="g/L",
        limit_type="internal_limit",
        notes="Test fixture",
        source_file="horizon_product_data.xlsx",
        source_type=SourceType.XLSX,
    )
