-- =============================================================================
-- Migration: 001_init_pgvector_hybrid.sql
-- Repository: cortex-rag-agent
-- Specification: .specs/02-data/hybrid-store-and-append-only-ledger.spec.md
-- Description:
--   1. Enables pgvector extension for dense vector similarity search.
--   2. Creates `document_chunks` with `vector(64)` HNSW index and generated
--      `tsvector` GIN index for hybrid BM25/FTS + Vector RRF retrieval.
--   3. Creates immutable append-only ledgers (`query_audit_log`, `eval_runs_ledger`)
--      guarded by `prevent_ledger_mutation()` BEFORE UPDATE / BEFORE DELETE triggers.
-- =============================================================================

BEGIN;

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Hybrid Knowledge Corpus Table (`document_chunks`)
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    title         TEXT NOT NULL,
    section       TEXT NOT NULL DEFAULT 'general',
    content       TEXT NOT NULL,
    tags_json     JSONB NOT NULL DEFAULT '[]'::jsonb,
    embedding     vector(64) NOT NULL,
    search_tsv    tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))
    ) STORED,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- HNSW index for fast approximate cosine similarity search over 64-dim unit embeddings
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for lexical full-text search over generated tsvector
CREATE INDEX IF NOT EXISTS idx_document_chunks_search_tsv_gin
    ON document_chunks
    USING gin (search_tsv);

-- Index on doc_id for fast document-level filtering
CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id
    ON document_chunks (doc_id);

-- 3. Append-Only Immutable Query Audit Ledger (`query_audit_log`)
CREATE TABLE IF NOT EXISTS query_audit_log (
    audit_id            TEXT PRIMARY KEY,
    query               TEXT NOT NULL,
    intent              TEXT NOT NULL,
    answer              TEXT NOT NULL,
    citations_json      JSONB NOT NULL DEFAULT '[]'::jsonb,
    faithfulness_score  DOUBLE PRECISION NOT NULL CHECK (faithfulness_score >= 0.0 AND faithfulness_score <= 1.0),
    latency_ms          DOUBLE PRECISION NOT NULL CHECK (latency_ms >= 0.0),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_audit_log_created_at
    ON query_audit_log (created_at DESC);

-- 4. Append-Only Immutable Evaluation Runs Ledger (`eval_runs_ledger`)
CREATE TABLE IF NOT EXISTS eval_runs_ledger (
    run_id                  TEXT PRIMARY KEY,
    total_cases             INTEGER NOT NULL CHECK (total_cases >= 0),
    passed_cases            INTEGER NOT NULL CHECK (passed_cases >= 0),
    avg_faithfulness        DOUBLE PRECISION NOT NULL,
    avg_citation_precision  DOUBLE PRECISION NOT NULL,
    avg_context_recall      DOUBLE PRECISION NOT NULL,
    avg_latency_ms          DOUBLE PRECISION NOT NULL,
    passed_gate             BOOLEAN NOT NULL,
    results_json            JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_ledger_created_at
    ON eval_runs_ledger (created_at DESC);

-- 5. Database-Level Append-Only Immutability Guard (`prevent_ledger_mutation`)
CREATE OR REPLACE FUNCTION prevent_ledger_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'APPEND_ONLY_LEDGER_VIOLATION: Operation % on immutable ledger table % is strictly prohibited',
        TG_OP, TG_TABLE_NAME
        USING ERRCODE = '23514';
END;
$$;

DROP TRIGGER IF EXISTS trg_query_audit_no_update ON query_audit_log;
CREATE TRIGGER trg_query_audit_no_update
    BEFORE UPDATE ON query_audit_log
    FOR EACH ROW
    EXECUTE FUNCTION prevent_ledger_mutation();

DROP TRIGGER IF EXISTS trg_query_audit_no_delete ON query_audit_log;
CREATE TRIGGER trg_query_audit_no_delete
    BEFORE DELETE ON query_audit_log
    FOR EACH ROW
    EXECUTE FUNCTION prevent_ledger_mutation();

DROP TRIGGER IF EXISTS trg_eval_ledger_no_update ON eval_runs_ledger;
CREATE TRIGGER trg_eval_ledger_no_update
    BEFORE UPDATE ON eval_runs_ledger
    FOR EACH ROW
    EXECUTE FUNCTION prevent_ledger_mutation();

DROP TRIGGER IF EXISTS trg_eval_ledger_no_delete ON eval_runs_ledger;
CREATE TRIGGER trg_eval_ledger_no_delete
    BEFORE DELETE ON eval_runs_ledger
    FOR EACH ROW
    EXECUTE FUNCTION prevent_ledger_mutation();

COMMIT;
