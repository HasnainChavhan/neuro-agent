"""
NeuroAgent — RAG Ingestion Pipeline
Chunks raw text, embeds it, and stores vectors in pgvector.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.rag.vector_store import vector_store
from app.utils.logger import get_logger

logger = get_logger("rag.pipeline")
settings = get_settings()


class RAGPipeline:
    """
    End-to-end RAG ingestion:
      raw text → chunk → embed → pgvector store

    Usage:
        pipeline = RAGPipeline()
        ids = await pipeline.ingest(text="...", source_url="https://...")
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = self._splitter.split_text(text)
        logger.debug("Chunked text", input_len=len(text), num_chunks=len(chunks))
        return chunks

    async def ingest(
        self,
        text: str,
        source_url: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> List[str]:
        """
        Ingest a single document.
        Returns list of document IDs created.
        """
        chunks = self.chunk(text)
        docs = [
            {
                "content": chunk,
                "source_url": source_url,
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "domain": urlparse(source_url).netloc if source_url else "",
                },
            }
            for i, chunk in enumerate(chunks)
        ]

        ids = await vector_store.upsert_batch(docs)
        logger.info(
            "Ingested document",
            source_url=source_url,
            chunks=len(chunks),
            stored=len(ids),
        )
        return ids

    async def ingest_many(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Ingest multiple documents from a list of {text, source_url, metadata}.
        """
        all_ids: List[str] = []
        for doc in documents:
            ids = await self.ingest(
                text=doc["text"],
                source_url=doc.get("source_url"),
                metadata=doc.get("metadata"),
            )
            all_ids.extend(ids)

        logger.info(
            "Batch ingestion complete",
            documents=len(documents),
            total_chunks=len(all_ids),
        )
        return all_ids

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_filter: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search wrapper."""
        return await vector_store.search(
            query=query,
            top_k=top_k or settings.rag_top_k,
            source_filter=source_filter,
        )


# Module-level singleton
rag_pipeline = RAGPipeline()
