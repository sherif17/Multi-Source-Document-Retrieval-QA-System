"""
Application configuration via environment variables.

Uses pydantic-settings for type-safe loading with validation.
Lazy singleton pattern: settings are only instantiated on first access
via `get_settings()`. This allows unit tests to import modules without
needing API keys set in the environment.

Architecture decision:
- All config in one place (no scattered os.getenv calls)
- Models configurable without code changes (swap gpt-4o-mini for gpt-4.1 via env)
- Defaults tuned for this project's scale (~200 rows, ~20 vectors)
- Lazy loading prevents import-time crashes in test environments
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API Keys ──
    openai_api_key: str
    pinecone_api_key: str
    neon_database_url: str

    # ── Model Configuration ──
    routing_model: str = "gpt-4o-mini"
    synthesis_model: str = "gpt-4.1"
    vision_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ── Pinecone ──
    pinecone_index_name: str = "doc-retrieval"

    # ── Retrieval Tuning ──
    vector_top_k: int = 5
    vector_score_threshold: float = 0.7
    max_retries: int = 2

    # ── Memory ──
    chat_history_window: int = 10

    # ── Paths ──
    docs_dir: Path = Path(__file__).parent.parent / "docs"
    data_dir: Path = Path(__file__).parent.parent / "data"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "extracted_images").mkdir(exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Lazy singleton — only instantiated on first call."""
    return Settings()


class _SettingsProxy:
    """
    Proxy that defers Settings instantiation until attribute access.
    Allows `from src.config import settings` without triggering validation
    at import time. Validation happens on first actual attribute read.
    """

    def __getattr__(self, name: str):
        return getattr(get_settings(), name)


settings = _SettingsProxy()
