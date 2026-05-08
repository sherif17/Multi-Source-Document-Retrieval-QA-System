"""
Prompts package — re-exports all prompt templates for backward-compatible imports.

Usage: `from src.prompts import ROUTER_SYSTEM, SYNTHESIS_USER, ...`
"""

from src.prompts.query_analyzer import QUERY_ANALYZER_SYSTEM, QUERY_ANALYZER_USER
from src.prompts.router import ROUTER_SYSTEM, ROUTER_USER
from src.prompts.sql_generation import SQL_GENERATION_SYSTEM, SQL_GENERATION_USER
from src.prompts.synthesis import (
    SYNTHESIS_RETRY_SUFFIX,
    SYNTHESIS_SYSTEM,
    SYNTHESIS_USER,
)

__all__ = [
    "QUERY_ANALYZER_SYSTEM",
    "QUERY_ANALYZER_USER",
    "ROUTER_SYSTEM",
    "ROUTER_USER",
    "SQL_GENERATION_SYSTEM",
    "SQL_GENERATION_USER",
    "SYNTHESIS_RETRY_SUFFIX",
    "SYNTHESIS_SYSTEM",
    "SYNTHESIS_USER",
]
