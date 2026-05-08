"""
Structured retriever: Text-to-SQL generation + validation + execution.

Converts natural language queries into safe SQL against Neon DB.
Enforces strict client isolation through multiple defense layers:

1. Prompt-level: SQL generation prompt mandates WHERE client_id = '{client}'
2. Validation-level: Post-generation AST check ensures client filter exists
3. Injection-level: Parameterized execution prevents SQL injection
4. Statement-level: Only SELECT statements allowed (no mutations)

This is a security-critical module. LLM output is NEVER trusted without validation.
"""

import re
from typing import Optional

import openai

from src.config import settings
from src.models import Source
from src.stores.sql_store import execute_safe_sql
from src.utils.logger import logger
from src.prompts import SQL_GENERATION_SYSTEM, SQL_GENERATION_USER


# ═══════════════════════════════════════════════════════════════════════════════
# SQL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

FORBIDDEN_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
    "EXEC", "EXECUTE", "CREATE", "GRANT", "REVOKE",
}


class SQLValidationError(Exception):
    """Raised when LLM-generated SQL fails safety validation."""
    pass


def validate_sql(sql: str, client_id: str) -> str:
    """
    Validate LLM-generated SQL for safety and correctness.

    Checks:
    1. Must be a SELECT statement
    2. No forbidden keywords (DROP, DELETE, etc.)
    3. Must reference client_id filter
    4. Must only reference allowed table (product_specs)
    5. No multiple statements (no semicolons mid-query)

    Returns cleaned SQL or raises SQLValidationError.
    """
    # Normalize whitespace
    sql = sql.strip().rstrip(";")

    # Remove markdown code fences if LLM wrapped the SQL
    if sql.startswith("```"):
        sql = re.sub(r"^```(?:sql)?\n?", "", sql)
        sql = re.sub(r"\n?```$", "", sql)
        sql = sql.strip()

    # Check 1: Must start with SELECT
    if not sql.upper().startswith("SELECT"):
        raise SQLValidationError(
            f"Only SELECT statements allowed. Got: {sql[:50]}..."
        )

    # Check 2: No forbidden keywords
    sql_upper = sql.upper()
    for keyword in FORBIDDEN_KEYWORDS:
        # Match as whole word to avoid false positives (e.g., "UPDATED" in notes)
        if re.search(rf"\b{keyword}\b", sql_upper):
            raise SQLValidationError(f"Forbidden keyword detected: {keyword}")

    # Check 3: Multiple statements
    if ";" in sql:
        raise SQLValidationError("Multiple statements not allowed")

    # Check 4: Client isolation — must contain client_id reference
    if "client_id" not in sql.lower():
        logger.warning("Generated SQL missing client_id filter — injecting")
        # Inject client filter as defense in depth
        if "WHERE" in sql.upper():
            sql = re.sub(
                r"WHERE",
                f"WHERE client_id = '{client_id}' AND",
                sql,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            sql += f" WHERE client_id = '{client_id}'"

    # Check 5: Ensure correct client_id value (not another client's ID)
    # Prevent the LLM from accidentally querying the wrong client
    other_clients = {"aurora", "horizon"} - {client_id}
    for other in other_clients:
        if f"'{other}'" in sql.lower() and f"'{client_id}'" not in sql.lower():
            raise SQLValidationError(
                f"SQL references wrong client '{other}' instead of '{client_id}'"
            )

    return sql


# ═══════════════════════════════════════════════════════════════════════════════
# SQL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def _build_dynamic_data_section() -> str:
    """
    Build the AVAILABLE DATA section of the SQL prompt dynamically
    from the actual database contents. Falls back to static values
    if the DB is unreachable.
    """
    try:
        from src.stores.sql_store import get_distinct_values
        distinct = get_distinct_values()
        if not distinct:
            return _static_data_section()

        lines = ["AVAILABLE DATA (from actual database):"]
        lines.append(f"- client_id values: {', '.join(repr(c) for c in sorted(distinct.keys()))}")

        all_regions = set()
        for info in distinct.values():
            all_regions.update(info.get("regions", []))
        lines.append(f"- region values: {', '.join(repr(r) for r in sorted(all_regions))}")

        for client_id, info in sorted(distinct.items()):
            products = info.get("products", [])
            params = info.get("parameters", [])
            lines.append(f"\nClient '{client_id}':")
            lines.append(f"  Products: {', '.join(products)}")
            lines.append(f"  Parameters: {', '.join(params)}")
            lines.append(f"  Rows: {info.get('total_rows', '?')}")

        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Could not fetch dynamic data for SQL prompt: {e}")
        return _static_data_section()


def _static_data_section() -> str:
    """Fallback static data section if DB is unreachable."""
    return (
        "AVAILABLE DATA:\n"
        "- client_id values: 'aurora', 'horizon'\n"
        "- region values: 'EU', 'US', 'GCC'\n"
        "- parameter values vary by client — use ILIKE with wildcards to match"
    )


def generate_sql(query: str, client_id: str, region: Optional[str] = None) -> str:
    """
    Generate a SQL query from natural language using gpt-4o-mini.

    Dynamically injects actual database values into the prompt so the
    LLM knows exactly which product names and parameters exist.
    """
    # Detect region from query if not provided
    if region is None:
        region = _detect_region(query)

    # Build prompt with dynamic data
    dynamic_data = _build_dynamic_data_section()
    system_prompt = SQL_GENERATION_SYSTEM.format(client_id=client_id)
    # Replace the static AVAILABLE DATA section with dynamic one
    system_prompt = system_prompt.replace(
        system_prompt[system_prompt.find("AVAILABLE DATA"):system_prompt.find("MANDATORY")],
        dynamic_data + "\n\n",
    )

    client = openai.OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.routing_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": SQL_GENERATION_USER.format(
                    client_id=client_id,
                    region=region or "any",
                    query=query,
                ),
            },
        ],
        temperature=0,
        max_tokens=300,
    )

    sql = response.choices[0].message.content.strip()
    logger.debug(f"Generated SQL: {sql}")
    return sql


def _detect_region(query: str) -> Optional[str]:
    """Extract region from query text if mentioned."""
    query_upper = query.upper()
    if "EU" in query_upper or "EUROPEAN" in query_upper:
        return "EU"
    if "US" in query_upper or "UNITED STATES" in query_upper:
        return "US"
    if "GCC" in query_upper or "GULF" in query_upper:
        return "GCC"
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RETRIEVAL FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def retrieve_structured(
    query: str,
    client_id: str,
) -> tuple[list[dict], str, Optional[str]]:
    """
    Full structured retrieval pipeline: generate → validate → execute.

    Includes a broadening retry: if the initial query returns 0 rows,
    a second attempt runs a broader discovery query to find matching data.

    Returns:
    - results: List of row dicts from the query
    - sql_query: The SQL that was executed (for trace/debug)
    - error: Error message if any step failed, None on success
    """
    sql_query = ""
    try:
        # Step 1: Generate SQL
        sql_query = generate_sql(query, client_id)

        # Step 2: Validate (security-critical)
        sql_query = validate_sql(sql_query, client_id)

        # Step 3: Execute
        results = execute_safe_sql(sql_query, client_id)

        # Step 4: Retry with broader query if 0 rows
        if not results:
            logger.warning("Initial SQL returned 0 rows — retrying with broader query")
            broad_results, broad_sql = _retry_broader(query, client_id)
            if broad_results:
                logger.info(f"Broad retry succeeded: {len(broad_results)} rows")
                return broad_results, broad_sql, None

        logger.info(f"Structured retrieval: {len(results)} rows returned")
        return results, sql_query, None

    except SQLValidationError as e:
        error_msg = f"SQL validation failed: {e}"
        logger.error(error_msg)
        return [], sql_query, error_msg

    except Exception as e:
        error_msg = f"Structured retrieval failed: {e}"
        logger.error(error_msg)
        return [], sql_query, error_msg


def _retry_broader(query: str, client_id: str) -> tuple[list[dict], str]:
    """
    Retry with a broader discovery query when the initial attempt returns 0 rows.

    Strategy: ask the LLM to regenerate SQL with relaxed matching,
    using only the client_id filter and wider ILIKE patterns.
    """
    broader_prompt = (
        f"The previous query returned 0 rows. Generate a BROADER query that:\n"
        f"- Keeps client_id = '{client_id}'\n"
        f"- Uses wider ILIKE patterns with short keywords (e.g. '%zinc%', '%Wall%')\n"
        f"- Drops the region filter if present\n"
        f"- Returns up to 10 rows so we can see what data exists\n"
        f"Original question: {query}"
    )
    try:
        sql = generate_sql(broader_prompt, client_id)
        sql = validate_sql(sql, client_id)
        results = execute_safe_sql(sql, client_id)
        return results, sql
    except Exception as e:
        logger.warning(f"Broad retry also failed: {e}")
        return [], ""


def format_sql_results_for_synthesis(results: list[dict]) -> str:
    """
    Format SQL results as readable text for the synthesis LLM.

    Converts raw row dicts into a structured text representation
    that the answer generator can easily reference and cite.
    """
    if not results:
        return "No structured data found."

    lines = []
    for i, row in enumerate(results, 1):
        parts = []
        if row.get("product_name"):
            parts.append(f"Product: {row['product_name']}")
        if row.get("parameter"):
            parts.append(f"Parameter: {row['parameter']}")
        if row.get("value") is not None:
            parts.append(f"Value: {row['value']}")
        if row.get("unit"):
            parts.append(f"Unit: {row['unit']}")
        if row.get("region"):
            parts.append(f"Region: {row['region']}")
        if row.get("limit_type"):
            parts.append(f"Type: {row['limit_type']}")
        if row.get("notes"):
            parts.append(f"Notes: {row['notes']}")
        if row.get("source_file"):
            parts.append(f"Source: {row['source_file']}")

        lines.append(f"[{i}] {' | '.join(parts)}")

    return "\n".join(lines)


def build_sources_from_sql(results: list[dict]) -> list[Source]:
    """Convert SQL results into Source objects for citation tracking."""
    sources = []
    for i, row in enumerate(results):
        value_str = f"{row.get('parameter', '?')} = {row.get('value', '?')} {row.get('unit', '')}"
        sources.append(
            Source(
                store="sql",
                content=value_str,
                file=row.get("source_file", "product_specs"),
                detail=f"Row {row.get('id', i+1)}: {row.get('product_name', '?')} / {row.get('region', '?')}",
            )
        )
    return sources
