"""
NeuroAgent — Test Suite: Agent Core
Tests query planning, orchestrator flow, retry logic, and mock tool behavior.
"""
from __future__ import annotations

import asyncio
import pytest

from app.agent.planner import QueryPlanner, ResearchPlan
from app.agent.orchestrator import AgentOrchestrator
from app.agent.retry_router import RetryRouter
from app.config import get_settings

settings = get_settings()


# ── Planner tests ─────────────────────────────────────────────────────────────

class TestQueryPlanner:
    planner = QueryPlanner()

    @pytest.mark.asyncio
    async def test_mock_plan_returns_sub_queries(self):
        plan = await self.planner.plan("What are the latest breakthroughs in quantum computing?")
        assert isinstance(plan, ResearchPlan)
        assert len(plan.sub_queries) >= 2
        assert plan.original_query

    @pytest.mark.asyncio
    async def test_plan_with_connector_splits(self):
        plan = await self.planner.plan("quantum computing and machine learning")
        assert len(plan.sub_queries) >= 1

    @pytest.mark.asyncio
    async def test_estimated_steps_positive(self):
        plan = await self.planner.plan("climate change impacts")
        assert plan.estimated_steps > 0

    @pytest.mark.asyncio
    async def test_plan_short_query(self):
        plan = await self.planner.plan("AI safety")
        assert plan.sub_queries


# ── Orchestrator tests ────────────────────────────────────────────────────────

class TestAgentOrchestrator:
    orchestrator = AgentOrchestrator()

    @pytest.mark.asyncio
    async def test_full_run_returns_report(self):
        report = await self.orchestrator.run("What is quantum entanglement?")
        assert report is not None
        assert report.task_id
        assert report.original_query == "What is quantum entanglement?"
        assert report.final_answer
        assert report.latency_ms > 0

    @pytest.mark.asyncio
    async def test_report_has_sources(self):
        report = await self.orchestrator.run("machine learning benchmarks 2024")
        # In mock mode, sources come from mock data
        assert isinstance(report.sources, list)

    @pytest.mark.asyncio
    async def test_report_has_sub_results(self):
        report = await self.orchestrator.run("What is RAG in AI?")
        assert len(report.sub_results) >= 1

    @pytest.mark.asyncio
    async def test_report_success_flag(self):
        report = await self.orchestrator.run("Explain transformer architecture")
        assert report.success is True

    @pytest.mark.asyncio
    async def test_steps_taken_positive(self):
        report = await self.orchestrator.run("What is CRISPR?")
        assert report.steps_taken > 0

    @pytest.mark.asyncio
    async def test_concurrent_runs(self):
        """Agent should handle multiple concurrent requests without deadlocking."""
        queries = [
            "quantum computing",
            "machine learning",
            "blockchain technology",
        ]
        reports = await asyncio.gather(
            *[self.orchestrator.run(q) for q in queries]
        )
        assert len(reports) == 3
        assert all(r.success for r in reports)


# ── Retry router tests ────────────────────────────────────────────────────────

class TestRetryRouter:
    router = RetryRouter(max_retries=3, backoff_base=0.01)  # fast backoff for tests

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        async def good_tool(x: int) -> int:
            return x * 2

        result = await self.router.call(good_tool, 5, tool_name="test_tool")
        assert result == 10

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        attempt_count = {"n": 0}

        async def flaky_tool() -> str:
            attempt_count["n"] += 1
            if attempt_count["n"] < 3:
                raise ConnectionError("Network failure")
            return "success"

        result = await self.router.call(flaky_tool, tool_name="flaky")
        assert result == "success"
        assert attempt_count["n"] == 3

    @pytest.mark.asyncio
    async def test_falls_back_after_exhaustion(self):
        async def always_fails() -> str:
            raise ConnectionError("Always fails")

        async def fallback() -> str:
            return "fallback_result"

        result = await self.router.call(
            always_fails,
            fallback=fallback,
            tool_name="exhausted_tool",
        )
        assert result == "fallback_result"

    @pytest.mark.asyncio
    async def test_raises_when_no_fallback(self):
        async def always_fails() -> str:
            raise ConnectionError("Always fails")

        with pytest.raises(RuntimeError, match="failed after"):
            await self.router.call(always_fails, tool_name="no_fallback")
