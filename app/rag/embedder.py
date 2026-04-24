"""
NeuroAgent — Text Embedder
Generates OpenAI text-embedding-3-small vectors with async batching.
Falls back to zero-vectors in mock mode.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import List

import openai

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("rag.embedder")
settings = get_settings()


class Embedder:
    """
    Async embedder with batch processing and mock fallback.

    In mock mode (MOCK_LLM=true), returns deterministic zero-vectors
    so the full pipeline can be tested without an OpenAI key.
    """

    MAX_BATCH = 64  # OpenAI batch limit

    def __init__(self) -> None:
        if not settings.mock_llm:
            self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        else:
            self._client = None
        self.model = settings.openai_embedding_model
        self.dim = settings.vector_dim

    async def embed_one(self, text: str) -> List[float]:
        """Embed a single text string."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings, chunking into MAX_BATCH-sized API calls.
        Returns embeddings in the same order as input.
        """
        if settings.mock_llm:
            return self._mock_embeddings(texts)

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.MAX_BATCH):
            chunk = texts[i : i + self.MAX_BATCH]
            response = await self._client.embeddings.create(
                model=self.model, input=chunk
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            logger.info(
                "Embedded batch",
                batch_start=i,
                batch_size=len(chunk),
                model=self.model,
            )

        return all_embeddings

    async def embed_parallel(self, texts: List[str]) -> List[List[float]]:
        """Parallel batching — faster for large corpora."""
        batches = [
            texts[i : i + self.MAX_BATCH] for i in range(0, len(texts), self.MAX_BATCH)
        ]
        results = await asyncio.gather(*[self.embed_batch(b) for b in batches])
        return [emb for batch in results for emb in batch]

    def _mock_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Deterministic mock embeddings based on text hash.
        Ensures cosine similarity tests are repeatable without an API key.
        """
        embeddings = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).hexdigest()
            # Build a pseudo-random but deterministic unit-ish vector
            seed = int(digest[:8], 16)
            vec = [(((seed * (i + 1)) % 997) / 997.0) for i in range(self.dim)]
            # Normalize
            norm = sum(x**2 for x in vec) ** 0.5 or 1.0
            embeddings.append([x / norm for x in vec])
        return embeddings

    @staticmethod
    def content_hash(text: str) -> str:
        """SHA-256 hash for deduplication."""
        return hashlib.sha256(text.encode()).hexdigest()


# Module-level singleton
embedder = Embedder()
