"""
NeuroAgent — Test Suite: RAG Pipeline
Tests embedder, vector store (mock), and pipeline ingestion.
"""
from __future__ import annotations

import pytest

from app.rag.embedder import Embedder
from app.rag.pipeline import RAGPipeline


# ── Embedder tests ──────────────────────────────────────────────────────────

class TestEmbedder:
    embedder = Embedder()

    @pytest.mark.asyncio
    async def test_embed_one_returns_correct_dim(self):
        vec = await self.embedder.embed_one("Hello, world!")
        assert len(vec) == 1536

    @pytest.mark.asyncio
    async def test_embed_batch_returns_all(self):
        texts = ["first", "second", "third"]
        vecs = await self.embedder.embed_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == 1536 for v in vecs)

    @pytest.mark.asyncio
    async def test_embed_parallel_consistent_with_batch(self):
        texts = [f"text {i}" for i in range(10)]
        batch = await self.embedder.embed_batch(texts)
        parallel = await self.embedder.embed_parallel(texts)
        # Same inputs → same mock vectors
        assert batch == parallel

    def test_content_hash_deterministic(self):
        h1 = self.embedder.content_hash("hello")
        h2 = self.embedder.content_hash("hello")
        assert h1 == h2

    def test_content_hash_different_inputs(self):
        h1 = self.embedder.content_hash("foo")
        h2 = self.embedder.content_hash("bar")
        assert h1 != h2

    @pytest.mark.asyncio
    async def test_mock_embeddings_normalized(self):
        """Mock vectors should have unit-ish norms."""
        vecs = await self.embedder.embed_batch(["test normalization"])
        vec = vecs[0]
        norm = sum(x**2 for x in vec) ** 0.5
        assert 0.99 < norm < 1.01, f"Norm out of range: {norm}"

    @pytest.mark.asyncio
    async def test_embed_empty_batch(self):
        vecs = await self.embedder.embed_batch([])
        assert vecs == []


# ── RAG Pipeline tests ───────────────────────────────────────────────────────

class TestRAGPipeline:
    pipeline = RAGPipeline(chunk_size=100, chunk_overlap=10)

    def test_chunk_splits_long_text(self):
        text = "word " * 200  # 1000 chars
        chunks = self.pipeline.chunk(text)
        assert len(chunks) > 1
        assert all(len(c) <= 120 for c in chunks)  # chunk_size + some tolerance

    def test_chunk_short_text_stays_single(self):
        text = "short text"
        chunks = self.pipeline.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    @pytest.mark.asyncio
    async def test_ingest_returns_ids(self, monkeypatch):
        """Patch vector_store.upsert_batch to avoid DB in tests."""
        from app.rag import pipeline as pipeline_module

        async def fake_upsert_batch(docs):
            return [f"id-{i}" for i in range(len(docs))]

        monkeypatch.setattr(
            pipeline_module.vector_store, "upsert_batch", fake_upsert_batch
        )

        ids = await self.pipeline.ingest(
            text="word " * 300,
            source_url="https://example.com/test",
        )
        assert len(ids) > 0
        assert all(id.startswith("id-") for id in ids)

    @pytest.mark.asyncio
    async def test_ingest_many_aggregates(self, monkeypatch):
        from app.rag import pipeline as pipeline_module

        async def fake_upsert_batch(docs):
            return [f"id-{i}" for i in range(len(docs))]

        monkeypatch.setattr(
            pipeline_module.vector_store, "upsert_batch", fake_upsert_batch
        )

        docs = [
            {"text": "word " * 100, "source_url": "https://a.com"},
            {"text": "word " * 100, "source_url": "https://b.com"},
        ]
        all_ids = await self.pipeline.ingest_many(docs)
        assert len(all_ids) > 1
