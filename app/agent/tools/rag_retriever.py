"""
NeuroAgent — RAG Retriever Tool
Retrieves semantically relevant documents from pgvector for a given query.
In mock mode, returns pre-cached synthetic documents.
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("agent.tools.rag_retriever")
settings = get_settings()


class RAGRetrieverTool:
    """
    Wraps the pgvector store for agent use.
    Handles DB-unavailable gracefully by returning empty results.
    """

    async def retrieve(
        self, query: str, top_k: int | None = None
    ) -> List[Dict[str, Any]]:
        """Returns top-k semantically similar documents."""
        if settings.mock_llm:
            return self._mock_documents(query, top_k or settings.rag_top_k)

        try:
            from app.rag.vector_store import vector_store

            results = await vector_store.search(
                query=query, top_k=top_k or settings.rag_top_k
            )
            logger.info(
                "RAG retrieval",
                query=query[:60],
                results=len(results),
            )
            return results
        except Exception as exc:
            logger.warning(
                "RAG retrieval failed, returning empty",
                error=str(exc),
                query=query[:60],
            )
            return []

    @staticmethod
    def _mock_documents(query: str, top_k: int) -> List[Dict[str, Any]]:
        """Synthetic documents for mock mode."""
        base = [
            {
                "id": f"mock-doc-{i}",
                "content": (
                    f"[Cached knowledge chunk {i+1} for query: '{query[:40]}']\n"
                    f"Research indicates that {query} involves multiple interconnected "
                    f"mechanisms. Prior work has established foundational principles "
                    f"that inform current understanding (similarity rank: {i+1})."
                ),
                "source_url": f"https://arxiv.org/abs/mock-{i:04d}",
                "metadata": {"chunk_index": i, "domain": "arxiv.org"},
                "similarity": round(0.95 - i * 0.05, 3),
            }
            for i in range(top_k)
        ]
        return base


# Module-level singleton
rag_retriever_tool = RAGRetrieverTool()
