-- NeuroAgent — PostgreSQL + pgvector Schema Migration 001
-- Run this against a fresh database before starting the application.

-- Enable the vector extension (requires pgvector installed)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Documents table ────────────────────────────────────────────────────────────
-- Stores chunked text content with 1536-dim OpenAI embeddings.
CREATE TABLE IF NOT EXISTS documents (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content       TEXT NOT NULL,
    embedding     vector(1536),
    metadata      JSONB DEFAULT '{}',
    content_hash  TEXT UNIQUE NOT NULL,
    source_url    TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- IVFFlat index for fast approximate cosine similarity search.
-- Rebuild with higher 'lists' value when document count exceeds 1M rows.
CREATE INDEX IF NOT EXISTS ix_documents_embedding_cosine
    ON documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- B-tree index on content_hash for deduplication lookups.
CREATE INDEX IF NOT EXISTS ix_documents_content_hash ON documents (content_hash);

-- ── Research sessions table ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS research_sessions (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query        TEXT NOT NULL,
    status       VARCHAR(20) NOT NULL DEFAULT 'pending',
    result       JSONB,
    error        TEXT,
    steps_taken  INTEGER NOT NULL DEFAULT 0,
    retries      INTEGER NOT NULL DEFAULT 0,
    tokens_used  INTEGER NOT NULL DEFAULT 0,
    latency_ms   FLOAT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Trigger: auto-update updated_at on row change
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_research_sessions_updated_at
    BEFORE UPDATE ON research_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ── Helper view ──────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW session_summary AS
SELECT
    status,
    COUNT(*)                            AS count,
    ROUND(AVG(latency_ms)::numeric, 2)  AS avg_latency_ms,
    ROUND(AVG(tokens_used)::numeric, 2) AS avg_tokens,
    MAX(created_at)                     AS last_run
FROM research_sessions
GROUP BY status;
