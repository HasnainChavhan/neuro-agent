# ADR 002 — Vector Database: pgvector

**Status:** Accepted  
**Date:** 2025-01-01  
**Deciders:** NeuroAgent Engineering

---

## Context

NeuroAgent's RAG pipeline requires a vector store for:
1. Storing 1536-dim OpenAI embeddings
2. Fast approximate nearest-neighbor (ANN) cosine similarity search
3. Metadata filtering (by source URL, domain, date)
4. ACID guarantees for upsert with deduplication

## Options Considered

| Solution | ANN Index | Metadata Filter | ACID | Infra Overhead | Cost |
|---|---|---|---|---|---|
| **pgvector** | IVFFlat / HNSW | JSONB queries | ✅ Full | Zero (existing Postgres) | $0 extra |
| Pinecone | HNSW | Namespaces | ❌ Eventual | SaaS, fully managed | $70+/mo |
| Weaviate | HNSW | GraphQL | ❌ Eventual | Separate service | $25+/mo |
| Chroma | HNSW | Metadata dict | ❌ Eventual | Embedded/separate | Free OSS |
| Qdrant | HNSW | Payload filters | ❌ Eventual | Separate service | Free OSS |

## Decision

**pgvector** (PostgreSQL extension) is chosen.

### Rationale

- **Zero additional infrastructure**: runs inside the existing PostgreSQL instance
- **ACID compliance**: upsert + deduplication (content_hash) works atomically
- **IVFFlat index**: achieves <10ms p99 search latency for <1M vectors
- **JSONB metadata**: full SQL expressiveness for filtering (source URL, domain, date range)
- **Operational simplicity**: one DB to monitor, backup, and scale

### Performance Characteristics

- IVFFlat with `lists=100` suitable for up to ~1M vectors
- Cosine similarity via `<=>` operator, natively supported
- For >10M vectors: upgrade to HNSW index (pgvector 0.5+), still within Postgres

### When to Migrate

Migrate to Pinecone/Qdrant if:
- Vector count exceeds 10M AND query latency > 50ms
- Multi-tenant namespacing at scale becomes critical
- GPU-accelerated ANN is required

## Consequences

- Tied to PostgreSQL (already a project dependency — no new service)
- Must manage index maintenance (`VACUUM`, `REINDEX`) at scale
- pgvector 0.6+ supports HNSW — upgrade path is straightforward
