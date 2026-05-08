"""
Tests for the query router (pre-router signals).

Tests the deterministic rule-based pre-router without requiring API calls.
LLM router tests would require OpenAI API and are in test_e2e.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.router import pre_route


class TestPreRouter:
    """Test deterministic signal extraction from queries."""

    def test_query1_aurora_single_client_numeric(self):
        query = "For client Aurora Paints, what is the maximum zinc content allowed in the finished EcoSafe Interior Wall Paint for the EU?"
        signals = pre_route(query)
        assert "aurora" in signals.extracted_clients
        assert signals.has_numeric_ask is True
        assert signals.has_narrative_ask is False
        assert signals.is_cross_client is False

    def test_query2_aurora_aggregation(self):
        query = "For client Aurora Paints, considering EcoSafe Ceiling Paint, EcoSafe Exterior Facade and EcoShield Floor Coating in the EU, what is the maximum internal VOC limit in g/L across these products?"
        signals = pre_route(query)
        assert "aurora" in signals.extracted_clients
        assert signals.has_numeric_ask is True
        assert signals.has_aggregation is True
        assert signals.has_product_list is True

    def test_query3_horizon_narrative(self):
        query = "According to the guidance for client Horizon Coatings' UltraSafe Interior Wall Paint, in which types of rooms is enhanced ventilation or longer airing-out periods specifically recommended?"
        signals = pre_route(query)
        assert "horizon" in signals.extracted_clients
        assert signals.has_narrative_ask is True
        assert signals.is_cross_client is False

    def test_query4_cross_client(self):
        query = "Comparing client Aurora Paints and client Horizon Coatings, which client sets a stricter VOC limit for interior wall paint in the EU?"
        signals = pre_route(query)
        assert "aurora" in signals.extracted_clients
        assert "horizon" in signals.extracted_clients
        assert signals.is_cross_client is True
        assert signals.has_numeric_ask is True

    def test_query5_aurora_hybrid(self):
        query = "For client Aurora Paints, what internal VOC limit in g/L is set for EcoSafe Kitchen & Bath in the EU for typical residential projects?"
        signals = pre_route(query)
        assert "aurora" in signals.extracted_clients
        assert signals.has_numeric_ask is True
        assert signals.is_cross_client is False

    def test_no_client_detected(self):
        query = "What is the maximum VOC content for interior wall paint?"
        signals = pre_route(query)
        assert signals.extracted_clients == ["unknown"]

    def test_comparison_keyword_triggers_cross_client(self):
        query = "Which client has stricter limits?"
        signals = pre_route(query)
        assert signals.is_cross_client is True
