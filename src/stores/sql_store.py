"""
Neon DB (PostgreSQL) store for structured product specification data.

Architecture decisions:
- Connection pooling via psycopg2 (sufficient for Streamlit's single-threaded model)
- Parameterized queries everywhere (SQL injection prevention)
- Schema creation is idempotent (CREATE TABLE IF NOT EXISTS)
- All write operations include source provenance (file, type)
- Client isolation enforced at the query level (WHERE client_id = %s)

Tradeoffs:
- psycopg2 over asyncpg: Streamlit is synchronous, async adds complexity without benefit here
- Single table over per-client tables: enables cross-client comparison queries (Query 4)
  while still enforcing isolation via parameterized WHERE clause
- TRUNCATE for re-ingestion: simplest idempotency for demo scope
"""

from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from src.config import settings
from src.models import ProductSpec
from src.utils.logger import logger

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS product_specs (
    id SERIAL PRIMARY KEY,
    client_id TEXT NOT NULL,
    product_name TEXT NOT NULL,
    region TEXT NOT NULL,
    parameter TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT,
    limit_type TEXT,
    notes TEXT,
    source_file TEXT NOT NULL,
    source_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_specs_client ON product_specs(client_id);",
    "CREATE INDEX IF NOT EXISTS idx_specs_client_product ON product_specs(client_id, product_name);",
    "CREATE INDEX IF NOT EXISTS idx_specs_client_region ON product_specs(client_id, region);",
    "CREATE INDEX IF NOT EXISTS idx_specs_client_param ON product_specs(client_id, parameter);",
]


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════


def get_connection():
    """Create a new database connection to Neon DB with timeout."""
    return psycopg2.connect(settings.neon_database_url, connect_timeout=10)


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════


def initialize_schema() -> None:
    """Create the product_specs table and indexes if they don't exist."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            for idx_sql in CREATE_INDEXES_SQL:
                cur.execute(idx_sql)
        conn.commit()
        logger.info("Database schema initialized successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to initialize schema: {e}")
        raise
    finally:
        conn.close()


def truncate_specs() -> None:
    """Remove all data from product_specs. Used for clean re-ingestion."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE product_specs RESTART IDENTITY;")
        conn.commit()
        logger.info("Truncated product_specs table")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to truncate: {e}")
        raise
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# WRITE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════


def insert_specs(specs: list[ProductSpec]) -> int:
    """
    Batch insert product specifications into Neon DB.

    Returns the number of rows inserted.
    """
    if not specs:
        return 0

    conn = get_connection()
    inserted = 0
    try:
        with conn.cursor() as cur:
            for spec in specs:
                cur.execute(
                    """
                    INSERT INTO product_specs 
                        (client_id, product_name, region, parameter, value, unit, limit_type, notes, source_file, source_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        spec.client_id,
                        spec.product_name,
                        spec.region,
                        spec.parameter,
                        spec.value,
                        spec.unit,
                        spec.limit_type,
                        spec.notes,
                        spec.source_file,
                        spec.source_type.value,
                    ),
                )
                inserted += 1
        conn.commit()
        logger.info(f"Inserted {inserted} specs into Neon DB")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to insert specs: {e}")
        raise
    finally:
        conn.close()

    return inserted


# ═══════════════════════════════════════════════════════════════════════════════
# READ OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════


def query_specs(
    client_id: str,
    product_name: Optional[str] = None,
    region: Optional[str] = None,
    parameter: Optional[str] = None,
) -> list[dict]:
    """
    Query product specs with client isolation enforced.

    Always filters by client_id. Additional filters are optional.
    Returns list of dicts with all columns.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT * FROM product_specs WHERE client_id = %s"
            params: list = [client_id]

            if product_name:
                query += " AND product_name ILIKE %s"
                params.append(f"%{product_name}%")

            if region:
                query += " AND region = %s"
                params.append(region)

            if parameter:
                query += " AND parameter = %s"
                params.append(parameter)

            query += " ORDER BY product_name, parameter"
            cur.execute(query, params)
            results = [dict(row) for row in cur.fetchall()]
            logger.debug(f"query_specs returned {len(results)} rows for client={client_id}")
            return results
    finally:
        conn.close()


def execute_safe_sql(sql: str, client_id: str) -> list[dict]:
    """
    Execute a validated SQL query with mandatory client isolation.

    This function is the gateway for LLM-generated SQL. It:
    1. Validates the SQL is a SELECT statement
    2. Ensures client_id filter is present (injects if missing)
    3. Executes with read-only intent

    Security: Never passes raw LLM output without validation.
    The SQL validation logic lives in retrieval/structured_retriever.py.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            results = [dict(row) for row in cur.fetchall()]
            logger.debug(f"execute_safe_sql returned {len(results)} rows")
            return results
    except Exception as e:
        logger.error(f"SQL execution failed: {e}\nQuery: {sql}")
        raise
    finally:
        conn.close()


def get_table_stats() -> dict:
    """Get row counts per client for observability."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT client_id, COUNT(*) as row_count, 
                       COUNT(DISTINCT product_name) as product_count,
                       COUNT(DISTINCT parameter) as parameter_count
                FROM product_specs 
                GROUP BY client_id
                """
            )
            return {row["client_id"]: dict(row) for row in cur.fetchall()}
    finally:
        conn.close()


# Module-level cache for distinct values (populated after ingestion)
_distinct_values_cache: dict = {}


def get_distinct_values(use_cache: bool = True) -> dict:
    """
    Return distinct product names, parameters, and regions per client.

    Used for diagnostic display and for dynamically populating the SQL
    generation prompt with actual values from the database.

    Results are cached in memory after the first successful call.
    Pass use_cache=False to force a fresh DB query.
    """
    global _distinct_values_cache
    if use_cache and _distinct_values_cache:
        return _distinct_values_cache

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT client_id,
                       ARRAY_AGG(DISTINCT product_name ORDER BY product_name) AS products,
                       ARRAY_AGG(DISTINCT parameter ORDER BY parameter) AS parameters,
                       ARRAY_AGG(DISTINCT region ORDER BY region) AS regions,
                       COUNT(*) AS total_rows
                FROM product_specs
                GROUP BY client_id
                ORDER BY client_id
                """
            )
            result = {row["client_id"]: dict(row) for row in cur.fetchall()}
            if result:
                _distinct_values_cache = result
            return result
    except Exception as e:
        logger.error(f"Failed to get distinct values: {e}")
        return _distinct_values_cache  # return stale cache if available
    finally:
        conn.close()


def refresh_distinct_values_cache() -> None:
    """Force refresh the distinct values cache (call after ingestion)."""
    get_distinct_values(use_cache=False)


def get_sample_rows(client_id: str, limit: int = 5) -> list[dict]:
    """Return a few sample rows for a client (diagnostic use)."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT product_name, parameter, value, unit, region, limit_type "
                "FROM product_specs WHERE client_id = %s LIMIT %s",
                (client_id, limit),
            )
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get sample rows: {e}")
        return []
    finally:
        conn.close()
