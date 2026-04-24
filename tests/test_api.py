"""
NeuroAgent — Test Suite: API Endpoints
Integration tests using httpx async client against the FastAPI app.
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    """Async test client for the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_schema(self, client):
        resp = await client.get("/api/v1/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "mock_mode" in data
        assert "metrics" in data

    @pytest.mark.asyncio
    async def test_root_redirect(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "NeuroAgent" in resp.json()["service"]


# ── Research endpoint ─────────────────────────────────────────────────────────

class TestResearch:
    @pytest.mark.asyncio
    async def test_research_returns_200(self, client):
        resp = await client.post(
            "/api/v1/research",
            json={"query": "What is quantum computing?"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_research_response_schema(self, client):
        resp = await client.post(
            "/api/v1/research",
            json={"query": "Explain transformer architecture"},
        )
        data = resp.json()
        assert "task_id" in data
        assert "answer" in data
        assert "sources" in data
        assert "latency_ms" in data
        assert "success" in data
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_research_answer_not_empty(self, client):
        resp = await client.post(
            "/api/v1/research",
            json={"query": "What is machine learning?"},
        )
        data = resp.json()
        assert len(data["answer"]) > 50

    @pytest.mark.asyncio
    async def test_research_has_sub_queries(self, client):
        resp = await client.post(
            "/api/v1/research",
            json={"query": "Climate change and renewable energy"},
        )
        data = resp.json()
        assert len(data["sub_queries"]) >= 1

    @pytest.mark.asyncio
    async def test_research_too_short_query(self, client):
        resp = await client.post(
            "/api/v1/research",
            json={"query": "AI"},  # < 5 chars
        )
        assert resp.status_code == 422  # Pydantic validation error

    @pytest.mark.asyncio
    async def test_research_missing_query(self, client):
        resp = await client.post("/api/v1/research", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_research_with_session_id(self, client):
        resp = await client.post(
            "/api/v1/research",
            json={
                "query": "What is CRISPR gene editing?",
                "session_id": "test-session-123",
            },
        )
        data = resp.json()
        assert data["task_id"] == "test-session-123"

    @pytest.mark.asyncio
    async def test_request_id_header(self, client):
        resp = await client.post(
            "/api/v1/research",
            json={"query": "Neural networks in medicine"},
            headers={"X-Request-ID": "my-custom-id"},
        )
        assert resp.headers.get("X-Request-ID") == "my-custom-id"

    @pytest.mark.asyncio
    async def test_latency_header_present(self, client):
        resp = await client.post(
            "/api/v1/research",
            json={"query": "History of the internet"},
        )
        assert "X-Latency-Ms" in resp.headers


# ── Metrics endpoint ──────────────────────────────────────────────────────────

class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_returns_summary(self, client):
        # Run a task first so metrics are non-zero
        await client.post(
            "/api/v1/research", json={"query": "What is deep learning?"}
        )
        resp = await client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks_started" in data
        assert "completion_rate" in data
