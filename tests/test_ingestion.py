"""
Tests for the ingestion pipeline modules.

Tests schema detection, normalization, content classification,
and spec extraction — all without requiring external API calls.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.content_classifier import classify_paragraph
from src.ingestion.spec_extractor import extract_specs_regex
from src.ingestion.xlsx_loader import detect_schema_variant, load_xlsx
from src.models import ContentType, Paragraph


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchemaDetection:
    def test_aurora_schema_detected(self):
        columns = ["client_name", "product_name", "region", "parameter", "value", "unit", "limit_type", "notes"]
        assert detect_schema_variant(columns) == "aurora"

    def test_horizon_schema_detected(self):
        columns = ["supplier_name", "product_line", "market", "metric", "metric_value", "metric_unit", "classification", "remarks"]
        assert detect_schema_variant(columns) == "horizon"

    def test_unknown_schema_raises(self):
        columns = ["foo", "bar", "baz"]
        with pytest.raises(ValueError, match="Cannot detect schema"):
            detect_schema_variant(columns)

    def test_case_insensitive_detection(self):
        columns = ["Client_Name", "Product_Name", "Region", "Parameter", "Value", "Unit", "Limit_Type", "Notes"]
        assert detect_schema_variant(columns) == "aurora"


# ═══════════════════════════════════════════════════════════════════════════════
# XLSX LOADING
# ═══════════════════════════════════════════════════════════════════════════════


class TestXlsxLoading:
    def test_aurora_xlsx_loads(self, aurora_xlsx_path):
        specs = load_xlsx(aurora_xlsx_path)
        assert len(specs) > 0
        assert all(s.client_id == "aurora" for s in specs)
        assert all(s.source_type.value == "xlsx" for s in specs)

    def test_horizon_xlsx_loads(self, horizon_xlsx_path):
        specs = load_xlsx(horizon_xlsx_path)
        assert len(specs) > 0
        assert all(s.client_id == "horizon" for s in specs)

    def test_aurora_regions_normalized(self, aurora_xlsx_path):
        specs = load_xlsx(aurora_xlsx_path)
        regions = {s.region for s in specs}
        assert regions.issubset({"EU", "US", "GCC"})

    def test_aurora_units_normalized(self, aurora_xlsx_path):
        specs = load_xlsx(aurora_xlsx_path)
        units = {s.unit for s in specs if s.unit}
        assert units.issubset({"g/L", "% by weight", "hours"})

    def test_horizon_has_more_parameters(self, horizon_xlsx_path):
        specs = load_xlsx(horizon_xlsx_path)
        params = {s.parameter for s in specs}
        # Horizon has zinc and drying time that Aurora doesn't
        assert "max_zinc_content" in params or "recommended_drying_time_before_occupancy" in params


# ═══════════════════════════════════════════════════════════════════════════════
# CONTENT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestContentClassification:
    def test_metadata_heading(self):
        para = Paragraph(index=0, text="Global Product Safety Brief – Client Aurora", style="Heading 1")
        assert classify_paragraph(para) == ContentType.METADATA

    def test_metadata_client_prefix(self):
        para = Paragraph(index=1, text="Client: Aurora Paints", style="Normal")
        assert classify_paragraph(para) == ContentType.METADATA

    def test_structured_voc_spec(self):
        para = Paragraph(
            index=4,
            text="Aurora Paints sets an internal maximum VOC content of 30 g/L",
            style="Normal",
        )
        assert classify_paragraph(para) == ContentType.STRUCTURED

    def test_structured_lead_spec(self):
        para = Paragraph(
            index=5,
            text="maximum lead content of 0.02 % by weight in the finished paint",
            style="Normal",
        )
        assert classify_paragraph(para) == ContentType.STRUCTURED

    def test_narrative_guidance(self):
        para = Paragraph(
            index=7,
            text="For children's rooms, nurseries and healthcare environments, local guidance recommends additional ventilation during application",
            style="Normal",
        )
        assert classify_paragraph(para) == ContentType.NARRATIVE

    def test_narrative_regulatory(self):
        para = Paragraph(
            index=6,
            text="Under current EU decorative coatings regulations, interior wall paints must comply with limits on hazardous substances",
            style="Normal",
        )
        assert classify_paragraph(para) == ContentType.NARRATIVE

    def test_ignore_bonus_note(self):
        para = Paragraph(
            index=15,
            text="Note: The image detection and text extraction is a bonus point not obligatory",
            style="Normal",
        )
        assert classify_paragraph(para) == ContentType.IGNORE

    def test_image_paragraph(self):
        para = Paragraph(index=16, text="", style="Normal", has_image=True)
        assert classify_paragraph(para) == ContentType.IMAGE


# ═══════════════════════════════════════════════════════════════════════════════
# SPEC EXTRACTION (REGEX)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpecExtraction:
    def test_extract_voc_limit(self):
        text = "For EcoSafe Interior Wall Paint in the EU, Aurora Paints sets an internal maximum VOC content of 30 g/L"
        specs = extract_specs_regex(text)
        assert len(specs) == 1
        assert specs[0].value == 30.0
        assert specs[0].unit == "g/L"
        assert specs[0].parameter == "max_voc_content"

    def test_extract_voc_with_product_and_region(self):
        text = "For EcoSafe Kitchen & Bath in the EU, Aurora Paints sets an internal VOC limit of 32 g/L"
        specs = extract_specs_regex(text)
        assert len(specs) == 1
        assert specs[0].value == 32.0
        assert specs[0].product_name is not None
        assert "Kitchen" in specs[0].product_name

    def test_extract_lead_content(self):
        text = "keeps UltraShield Primer at a maximum lead content of 0.02 % by weight"
        specs = extract_specs_regex(text)
        assert len(specs) == 1
        assert specs[0].value == 0.02
        assert specs[0].parameter == "max_lead_content"
        assert specs[0].unit == "% by weight"

    def test_no_specs_in_narrative(self):
        text = "local guidance recommends additional ventilation during application and minimum drying times"
        specs = extract_specs_regex(text)
        assert len(specs) == 0

    def test_extract_horizon_voc(self):
        text = "For UltraSafe Interior Wall Paint in the EU, Horizon Coatings applies an internal VOC threshold of 25 g/L"
        specs = extract_specs_regex(text)
        assert len(specs) == 1
        assert specs[0].value == 25.0
        assert specs[0].region == "EU"
