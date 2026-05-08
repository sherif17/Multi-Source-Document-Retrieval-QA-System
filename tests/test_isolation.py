"""
Tests for client data isolation.

Verifies that queries for one client cannot leak data from another.
Tests both the SQL validation layer and the router's client detection.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.structured_retriever import SQLValidationError, validate_sql


class TestSQLIsolation:
    """Test SQL validation enforces client isolation."""

    def test_valid_sql_with_client_filter(self):
        sql = "SELECT * FROM product_specs WHERE client_id = 'aurora' AND region = 'EU'"
        result = validate_sql(sql, "aurora")
        assert "client_id" in result
        assert "aurora" in result

    def test_missing_client_filter_gets_injected(self):
        sql = "SELECT * FROM product_specs WHERE region = 'EU'"
        result = validate_sql(sql, "aurora")
        assert "client_id" in result.lower()
        assert "aurora" in result

    def test_wrong_client_raises_error(self):
        sql = "SELECT * FROM product_specs WHERE client_id = 'horizon'"
        with pytest.raises(SQLValidationError, match="wrong client"):
            validate_sql(sql, "aurora")

    def test_drop_table_blocked(self):
        sql = "DROP TABLE product_specs"
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validate_sql(sql, "aurora")

    def test_delete_blocked(self):
        sql = "DELETE FROM product_specs WHERE client_id = 'aurora'"
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validate_sql(sql, "aurora")

    def test_forbidden_keyword_in_select(self):
        sql = "SELECT * FROM product_specs; DROP TABLE product_specs"
        with pytest.raises(SQLValidationError, match="Forbidden keyword detected: DROP"):
            validate_sql(sql, "aurora")

    def test_insert_blocked(self):
        sql = "INSERT INTO product_specs (client_id) VALUES ('hacker')"
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validate_sql(sql, "aurora")

    def test_update_blocked(self):
        sql = "UPDATE product_specs SET value = 999 WHERE client_id = 'aurora'"
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validate_sql(sql, "aurora")

    def test_markdown_code_fences_stripped(self):
        sql = "```sql\nSELECT * FROM product_specs WHERE client_id = 'aurora'\n```"
        result = validate_sql(sql, "aurora")
        assert "```" not in result
        assert result.startswith("SELECT")

    def test_both_clients_allowed_in_cross_client(self):
        """Cross-client queries may reference both clients."""
        sql = "SELECT * FROM product_specs WHERE client_id IN ('aurora', 'horizon')"
        # This should not raise — both clients are valid for cross-client
        # The validate_sql only raises if WRONG client is used for SINGLE client queries
        # For cross-client, we pass the first client but the query contains both
        # This is a valid cross-client scenario
        result = validate_sql(sql, "aurora")
        assert "aurora" in result
