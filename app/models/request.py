"""
NeuroAgent — API Request Schemas
"""
from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class ResearchRequest(BaseModel):
    """Input schema for POST /research"""

    query: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="The research question or topic to investigate",
        examples=["What are the latest breakthroughs in quantum computing?"],
    )
    session_id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()),
        description="Optional session ID for request tracing. Auto-generated if not provided.",
    )
    max_results: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results per sub-query",
    )
    top_k_rag: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of RAG documents to retrieve per sub-query",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class IngestRequest(BaseModel):
    """Input schema for POST /ingest — add documents to the knowledge base"""

    text: str = Field(..., min_length=10, description="Raw text to ingest")
    source_url: Optional[str] = Field(None, description="Source URL for citation")
    metadata: Optional[dict] = Field(default_factory=dict, description="Arbitrary metadata")
