"""
NeuroAgent — Agent Orchestrator
End-to-end autonomous research pipeline:
  query → plan → [search + retrieve] × N sub-queries → synthesize → report
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.agent.planner import ResearchPlan, query_planner
from app.agent.retry_router import retry_router
from app.agent.tools.rag_retriever import rag_retriever_tool
from app.agent.tools.synthesizer import synthesizer_tool
from app.agent.tools.web_search import WebSearchResult, web_search_tool
from app.config import get_settings
from app.utils.logger import BoundLogger, get_logger
from app.utils.metrics import metrics_store

logger = get_logger("agent.orchestrator")
settings = get_settings()


@dataclass
class SubQueryResult:
    sub_query: str
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    rag_docs: List[Dict[str, Any]] = field(default_factory=list)
    synthesis: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


@dataclass
class ResearchReport:
    task_id: str
    original_query: str
    plan: ResearchPlan
    sub_results: List[SubQueryResult]
    final_answer: str
    sources: List[str]
    tokens_used: int
    latency_ms: float
    steps_taken: int
    retries: int
    fallbacks: int
    success: bool = True


class AgentOrchestrator:
    """
    Autonomous multi-step research agent.

    Pipeline per sub-query:
      1. WebSearchTool  → live web results
      2. RAGRetrieverTool → cached vector store knowledge
      3. SynthesizerTool → GPT-4o structured answer

    Retry logic wraps each tool call via RetryRouter.
    All sub-queries run concurrently (asyncio.gather) for speed.
    Results are merged into a unified ResearchReport.
    """

    async def run(self, query: str, session_id: str | None = None) -> ResearchReport:
        task_id = session_id or str(uuid.uuid4())
        metrics = metrics_store.start_task(task_id, query)
        log = BoundLogger(logger, task_id=task_id, query=query[:60])

        log.info("Agent task started")
        start = time.monotonic()

        try:
            # ── Step 1: Decompose query ─────────────────────────────────────
            plan = await asyncio.wait_for(
                query_planner.plan(query),
                timeout=settings.task_timeout_seconds,
            )
            metrics.sub_queries = plan.sub_queries
            log.info("Plan ready", sub_queries=len(plan.sub_queries))

            # ── Step 2: Execute sub-queries concurrently ─────────────────────
            sub_results = await asyncio.gather(
                *[
                    self._execute_sub_query(sq, task_id=task_id, log=log)
                    for sq in plan.sub_queries
                ],
                return_exceptions=False,
            )

            # ── Step 3: Merge results ───────────────────────────────────────
            all_search_results: List[Dict[str, Any]] = []
            all_rag_docs: List[Dict[str, Any]] = []
            all_sources: List[str] = []
            total_tokens = 0

            for sr in sub_results:
                all_search_results.extend(sr.search_results)
                all_rag_docs.extend(sr.rag_docs)
                if sr.synthesis.get("sources"):
                    all_sources.extend(sr.synthesis["sources"])
                total_tokens += sr.synthesis.get("tokens_used", 0)

            # ── Step 4: Final synthesis over all collected evidence ──────────
            final = await retry_router.call(
                synthesizer_tool.synthesize,
                query,
                all_search_results,
                all_rag_docs,
                tool_name="final_synthesizer",
                task_id=task_id,
            )

            latency_ms = (time.monotonic() - start) * 1000
            steps_taken = len(plan.sub_queries) * 3 + 1  # search+rag+synth × N + final

            task_metrics = metrics_store.get_task(task_id)
            metrics_store.finish_task(task_id, success=True)

            report = ResearchReport(
                task_id=task_id,
                original_query=query,
                plan=plan,
                sub_results=list(sub_results),
                final_answer=final.get("answer", ""),
                sources=list(dict.fromkeys(all_sources + (final.get("sources") or []))),
                tokens_used=total_tokens + final.get("tokens_used", 0),
                latency_ms=round(latency_ms, 1),
                steps_taken=steps_taken,
                retries=task_metrics.retry_events if task_metrics else 0,
                fallbacks=task_metrics.fallback_events if task_metrics else 0,
                success=True,
            )

            log.info(
                "Agent task complete",
                latency_ms=report.latency_ms,
                steps=steps_taken,
                sources=len(report.sources),
            )
            return report

        except Exception as exc:
            metrics_store.finish_task(task_id, success=False)
            latency_ms = (time.monotonic() - start) * 1000
            log.error("Agent task failed", error=str(exc))

            # Return a partial report so the API always responds
            return ResearchReport(
                task_id=task_id,
                original_query=query,
                plan=ResearchPlan(query, [query], "fallback"),
                sub_results=[],
                final_answer=f"Research could not be completed: {str(exc)}",
                sources=[],
                tokens_used=0,
                latency_ms=round(latency_ms, 1),
                steps_taken=0,
                retries=0,
                fallbacks=0,
                success=False,
            )

    async def _execute_sub_query(
        self,
        sub_query: str,
        task_id: str,
        log: BoundLogger,
    ) -> SubQueryResult:
        """Run search + RAG + synthesis for a single sub-query."""
        result = SubQueryResult(sub_query=sub_query)

        # ── Web search (with RAG as fallback) ───────────────────────────────
        search_results_raw = await retry_router.call(
            web_search_tool.search,
            sub_query,
            fallback=lambda q, **kw: rag_retriever_tool.retrieve(q),
            tool_name="web_search",
            task_id=task_id,
        )
        # Convert WebSearchResult objects to dicts if needed
        result.search_results = [
            r.to_dict() if isinstance(r, WebSearchResult) else r
            for r in (search_results_raw or [])
        ]

        # ── RAG retrieval ────────────────────────────────────────────────────
        result.rag_docs = await retry_router.call(
            rag_retriever_tool.retrieve,
            sub_query,
            tool_name="rag_retriever",
            task_id=task_id,
        ) or []

        # ── Sub-query synthesis ──────────────────────────────────────────────
        result.synthesis = await retry_router.call(
            synthesizer_tool.synthesize,
            sub_query,
            result.search_results,
            result.rag_docs,
            tool_name="synthesizer",
            task_id=task_id,
        )

        task = metrics_store.get_task(task_id)
        if task:
            task.tools_invoked.extend(["web_search", "rag_retriever", "synthesizer"])

        log.info(
            "Sub-query done",
            sub_query=sub_query[:50],
            search=len(result.search_results),
            rag=len(result.rag_docs),
        )
        return result


# Module-level singleton
orchestrator = AgentOrchestrator()
