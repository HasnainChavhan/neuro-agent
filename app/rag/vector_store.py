"""
NeuroAgent — pgvector Store
Async CRUD operations for document embeddings with cosine similarity search.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.models import Document
from app.db.session import get_session
from app.rag.embedder import embedder
from app.utils.logger import get_logger

logger = get_logger("rag.vector_store")


class VectorStore:
    """
    pgvector-backed async vector store.

    Provides:
    - Upsert with deduplication (content_hash)
    - Cosine similarity search
    - Batch ingestion
    - Namespace/source filtering via metadata JSONB
    """

    async def upsert(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        source_url: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        """Insert or update a document. Returns document ID."""
        if embedding is None:
            embedding = await embedder.embed_one(content)

        content_hash = embedder.content_hash(content)

        stmt = (
            pg_insert(Document)
            .values(
                content=content,
                embedding=embedding,
                content_hash=content_hash,
                source_url=source_url,
                metadata_=metadata or {},
            )
            .on_conflict_do_update(
                index_elements=["content_hash"],
                set_={
                    "embedding": embedding,
                    "metadata_": metadata or {},
                },
            )
            .returning(Document.id)
        )

        async with get_session() as session:
            result = await session.execute(stmt)
            doc_id = result.scalar_one()

        logger.info("Upserted document", doc_id=str(doc_id), source_url=source_url)
        return str(doc_id)

    async def upsert_batch(
        self,
        docs: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Batch upsert. Each dict: {content, source_url?, metadata?}
        Embeds all in parallel then upserts in a single transaction.
        """
        texts = [d["content"] for d in docs]
        embeddings = await embedder.embed_parallel(texts)

        ids: List[str] = []
        async with get_session() as session:
            for doc, emb in zip(docs, embeddings):
                content_hash = embedder.content_hash(doc["content"])
                stmt = (
                    pg_insert(Document)
                    .values(
                        content=doc["content"],
                        embedding=emb,
                        content_hash=content_hash,
                        source_url=doc.get("source_url"),
                        metadata_=doc.get("metadata", {}),
                    )
                    .on_conflict_do_update(
                        index_elements=["content_hash"],
                        set_={"embedding": emb},
                    )
                    .returning(Document.id)
                )
                result = await session.execute(stmt)
                ids.append(str(result.scalar_one()))

        logger.info("Batch upserted documents", count=len(ids))
        return ids

    async def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Cosine similarity search. Returns top-k ranked documents.
        Optionally filters by source_url prefix.
        """
        query_emb = await embedder.embed_one(query)
        emb_str = "[" + ",".join(str(x) for x in query_emb) + "]"

        async with get_session() as session:
            sql = text(
                """
                SELECT
                    id,
                    content,
                    source_url,
                    metadata,
                    1 - (embedding <=> :embedding::vector) AS similarity
                FROM documents
                WHERE (:source_filter IS NULL OR source_url LIKE :source_filter)
                ORDER BY embedding <=> :embedding::vector
                LIMIT :top_k
                """
            )
            result = await session.execute(
                sql,
                {
                    "embedding": emb_str,
                    "top_k": top_k,
                    "source_filter": f"{source_filter}%" if source_filter else None,
                },
            )
            rows = result.mappings().all()

        results = [
            {
                "id": str(row["id"]),
                "content": row["content"],
                "source_url": row["source_url"],
                "metadata": row["metadata"],
                "similarity": float(row["similarity"]),
            }
            for row in rows
        ]

        logger.info(
            "Vector search complete",
            query=query[:60],
            results=len(results),
            top_similarity=results[0]["similarity"] if results else 0,
        )
        return results

    async def delete_by_source(self, source_url: str) -> int:
        """Delete all documents from a given source URL. Returns count deleted."""
        async with get_session() as session:
            stmt = delete(Document).where(Document.source_url == source_url)
            result = await session.execute(stmt)
            return result.rowcount

    async def count(self) -> int:
        """Total number of documents in the store."""
        async with get_session() as session:
            result = await session.execute(select(Document.id))
            return len(result.all())


# Module-level singleton
vector_store = VectorStore()
