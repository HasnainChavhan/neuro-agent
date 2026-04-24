"""
NeuroAgent — Application Configuration
Centralizes all settings loaded from environment variables / .env file.
"""
from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ──────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="sk-mock-key", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    mock_llm: bool = Field(default=True, alias="MOCK_LLM")

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://neuro:neuro_secret@localhost:5432/neuro_agent",
        alias="DATABASE_URL",
    )
    database_pool_size: int = Field(default=10, alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, alias="DATABASE_MAX_OVERFLOW")

    # ── Agent ─────────────────────────────────────────────────────────────────
    max_agent_steps: int = Field(default=10, alias="MAX_AGENT_STEPS")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    retry_backoff_base: float = Field(default=2.0, alias="RETRY_BACKOFF_BASE")
    task_timeout_seconds: int = Field(default=120, alias="TASK_TIMEOUT_SECONDS")

    # ── RAG ───────────────────────────────────────────────────────────────────
    vector_dim: int = Field(default=1536, alias="VECTOR_DIM")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")

    # ── Search ────────────────────────────────────────────────────────────────
    search_provider: str = Field(default="duckduckgo", alias="SEARCH_PROVIDER")
    serpapi_key: str = Field(default="", alias="SERPAPI_KEY")
    brave_api_key: str = Field(default="", alias="BRAVE_API_KEY")

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    workers: int = Field(default=1, alias="WORKERS")
    log_level: str = Field(default="info", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a cached singleton Settings instance."""
    return Settings()
