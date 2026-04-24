"""
NeuroAgent — Research Route
POST /research  — Main agent endpoint
POST /ingest    — Add documents to the knowledge base
GET  /metrics   — Runtime metrics
"""
from __future__ import annotations

import traceback
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.agent.orchestrator import orchestrator
from app.models.request import IngestRequest, ResearchRequest
from app.models.response import (
    ErrorResponse,
    IngestResponse,
    ResearchReport,
    SubQuerySummary,
)
from app.rag.pipeline import rag_pipeline
from app.utils.logger import get_logger
from app.utils.metrics import metrics_store

router = APIRouter()
logger = get_logger("api.research")


@router.post(
    "/research",
    response_model=ResearchReport,
    status_code=status.HTTP_200_OK,
    summary="Run Autonomous Research Agent",
    description=(
        "Decomposes the query, executes multi-step web search + RAG retrieval, "
        "synthesizes results, and returns a structured Markdown report."
    ),
    tags=["Research Agent"],
    responses={
        500: {"model": ErrorResponse, "description": "Agent execution error"},
    },
)
async def run_research(request: ResearchRequest) -> ResearchReport:
    logger.info(
        "Research request received",
        session_id=request.session_id,
        query=request.query[:80],
    )

    try:
        report = await orchestrator.run(
            query=request.query,
            session_id=request.session_id,
        )
    except Exception as exc:
        logger.error("Unhandled error in research endpoint", error=str(exc))
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(exc)}",
        )

    # Map domain model → response schema
    sub_summaries: List[SubQuerySummary] = [
        SubQuerySummary(
            sub_query=sr.sub_query,
            search_results_count=len(sr.search_results),
            rag_docs_count=len(sr.rag_docs),
            success=sr.success,
        )
        for sr in report.sub_results
    ]

    return ResearchReport(
        task_id=report.task_id,
        query=report.original_query,
        answer=report.final_answer,
        sources=report.sources,
        sub_queries=report.plan.sub_queries,
        sub_results=sub_summaries,
        metadata={
            "reasoning": report.plan.reasoning,
            "model": "mock" if True else "gpt-4o",
        },
        latency_ms=report.latency_ms,
        steps_taken=report.steps_taken,
        retries=report.retries,
        fallbacks=report.fallbacks,
        tokens_used=report.tokens_used,
        success=report.success,
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Document into Knowledge Base",
    tags=["Knowledge Base"],
)
async def ingest_document(request: IngestRequest) -> IngestResponse:
    """Chunks and embeds a document into the pgvector store."""
    try:
        ids = await rag_pipeline.ingest(
            text=request.text,
            source_url=request.source_url,
            metadata=request.metadata,
        )
        return IngestResponse(
            ingested_chunks=len(ids),
            document_ids=ids,
            source_url=request.source_url,
        )
    except Exception as exc:
        logger.error("Ingest failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(exc)}",
        )


@router.get(
    "/metrics",
    summary="Runtime Metrics",
    tags=["Infrastructure"],
)
async def get_metrics() -> dict:
    """Returns agent task completion rates and retry statistics."""
    return metrics_store.summary()
