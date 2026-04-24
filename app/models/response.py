"""
NeuroAgent — API Response Schemas
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SubQuerySummary(BaseModel):
    sub_query: str
    search_results_count: int
    rag_docs_count: int
    success: bool


class ResearchReport(BaseModel):
    """Full structured research report returned by POST /research"""

    task_id: str = Field(..., description="Unique ID for this research task")
    query: str = Field(..., description="Original research question")
    answer: str = Field(..., description="Synthesized Markdown answer")
    sources: List[str] = Field(default_factory=list, description="Cited source URLs")
    sub_queries: List[str] = Field(
        default_factory=list, description="Decomposed sub-questions"
    )
    sub_results: List[SubQuerySummary] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # ── Performance telemetry ─────────────────────────────────────────────
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds")
    steps_taken: int = Field(..., description="Total agent steps executed")
    retries: int = Field(default=0, description="Number of retry events")
    fallbacks: int = Field(default=0, description="Number of fallback routing events")
    tokens_used: int = Field(default=0, description="Total LLM tokens consumed")
    success: bool = Field(..., description="Whether the task completed successfully")

    model_config = {"json_schema_extra": {"example": {
        "task_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        "query": "What are the latest breakthroughs in quantum computing?",
        "answer": "## Answer\n\nQuantum computing has seen remarkable progress...",
        "sources": ["https://nature.com/...", "https://arxiv.org/..."],
        "sub_queries": ["Overview of quantum computing", "Recent breakthroughs"],
        "latency_ms": 312.4,
        "steps_taken": 7,
        "retries": 0,
        "fallbacks": 0,
        "tokens_used": 842,
        "success": True,
    }}}


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    mock_mode: bool
    metrics: Dict[str, Any]


class IngestResponse(BaseModel):
    ingested_chunks: int
    document_ids: List[str]
    source_url: Optional[str]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    task_id: Optional[str] = None
