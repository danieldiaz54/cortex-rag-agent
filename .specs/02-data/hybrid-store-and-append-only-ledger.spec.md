# SPEC-02: Hybrid Vector/Lexical Storage & Append-Only Audit/Evaluation Ledgers

- **Bounded Context**: `cortex-rag-agent` — Persistence & Data Architecture (`src/infrastructure/persistence/`, `migrations/`)
- **Status**: Approved (SDD-First)
- **Author Role**: `data-architect`
- **Target Environments**:
  - **Development / CI (`CORTEX_ENV=development`)**: Local SQLite (`SQLiteHybridStore`) or isolated Dockerized PostgreSQL 16 + `pgvector` (`docker-compose.dev.yml` / `.env.development`).
  - **Production (`CORTEX_ENV=production`)**: Managed PostgreSQL 16+ with `pgvector` HNSW indexing and `tsvector` GIN full-text search (`.env.production`).

---

## 1. Architectural Objectives & Data Governance Rules

1. **Hybrid Storage Contract (`document_chunks`)**:
   - Persist technical documentation chunks with structured metadata (`chunk_id`, `doc_id`, `title`, `section`, `tags`) alongside 64-dimensional L2-normalized dense embeddings (`vector(64)` in PostgreSQL / JSON float array in SQLite).
   - Enable deterministic enrichment via `compute_deterministic_embedding` when a `DocumentChunk` is upserted without precomputed coordinates.
2. **Append-Only Ledger Immutability (`query_audit_log` & `eval_runs_ledger`)**:
   - Every RAG execution (`RagResponse`) and Golden Dataset evaluation run (`EvalSuiteReport`) is recorded in an **immutable, append-only ledger**.
   - **Database-Level Trigger Enforcement**: Both SQLite (dev/test) and PostgreSQL (staging/prod) enforce `BEFORE UPDATE` and `BEFORE DELETE` triggers on `query_audit_log` and `eval_runs_ledger`. Any attempt to mutate or delete historical audit or evaluation records aborts the transaction with error code/message `APPEND_ONLY_LEDGER_VIOLATION` (surfaced in Python as `AppendOnlyLedgerError`).
3. **Strict Environment Isolation**:
   - Development runs exclusively against `.env.development` (`CORTEX_SQLITE_PATH` or local container `localhost:5439/cortex_rag_dev`).
   - Production credentials and endpoints live strictly in `.env.production` (never committed, never referenced from `dev` branches or test suites).
4. **Repository Inversion**:
   - `SQLiteHybridStore` implements the domain repository protocols (`ChunkRepositoryProtocol` and `LedgerRepositoryProtocol`) and returns pure domain entities (`DocumentChunk`) without leaking `sqlite3.Row` or SQL cursors into `src/domain/`.

---

## 2. Relational & Vector Schema Design

```mermaid
erDiagram
    DOCUMENT_CHUNKS {
        text chunk_id PK
        text doc_id
        text title
        text section
        text content
        jsonb tags_json
        vector_64 embedding
        tsvector search_tsv
        timestamptz created_at
    }

    QUERY_AUDIT_LOG {
        text audit_id PK
        text query
        text intent
        text answer
        jsonb citations_json
        double faithfulness_score
        double latency_ms
        timestamptz created_at
    }

    EVAL_RUNS_LEDGER {
        text run_id PK
        int total_cases
        int passed_cases
        double avg_faithfulness
        double avg_citation_precision
        double avg_context_recall
        double avg_latency_ms
        boolean passed_gate
        jsonb results_json
        timestamptz created_at
    }
```

### 2.1 `document_chunks` (Mutable Knowledge Corpus)
- **Primary Key**: `chunk_id` (`TEXT`).
- **Columns**:
  - `doc_id` (`TEXT NOT NULL`): Logical parent document identifier (e.g., `DOC-DIST-001`).
  - `title` (`TEXT NOT NULL`): Document or section title.
  - `section` (`TEXT NOT NULL`): Structural section header.
  - `content` (`TEXT NOT NULL`): Full technical text passage used for BM25 and dense vector indexing.
  - `tags` / `tags_json` (`JSONB` in Postgres, `TEXT` JSON array in SQLite): Domain tags (`["bm25", "pgvector", "rrf"]`).
  - `embedding` / `embedding_json` (`vector(64)` in Postgres with `hnsw (embedding vector_cosine_ops)`, `TEXT` JSON float array in SQLite).
  - `search_tsv` (PostgreSQL generated column): `GENERATED ALWAYS AS (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,''))) STORED`, indexed via `GIN`.
  - `created_at` (`TIMESTAMPTZ` / ISO-8601 `TEXT`).
- **Upsert Semantics**: `INSERT ... ON CONFLICT(chunk_id) DO UPDATE` refreshes corpus chunks idempotently during ingestion pipelines.

### 2.2 `query_audit_log` (Append-Only Immutable Audit Ledger)
- **Primary Key**: `audit_id` (`TEXT`).
- **Columns**:
  - `query` (`TEXT NOT NULL`)
  - `intent` (`TEXT NOT NULL`)
  - `answer` (`TEXT NOT NULL`)
  - `citations_json` (`TEXT` / `JSONB NOT NULL`)
  - `faithfulness_score` (`REAL` / `DOUBLE PRECISION NOT NULL`)
  - `latency_ms` (`REAL` / `DOUBLE PRECISION NOT NULL`)
  - `created_at` (`TEXT` / `TIMESTAMPTZ NOT NULL`)
- **Immutability Triggers**:
  - `trg_query_audit_no_update`: Fires `BEFORE UPDATE ON query_audit_log`, aborts with `APPEND_ONLY_LEDGER_VIOLATION: Ledger records are immutable`.
  - `trg_query_audit_no_delete`: Fires `BEFORE DELETE ON query_audit_log`, aborts with `APPEND_ONLY_LEDGER_VIOLATION: Ledger records cannot be deleted`.

### 2.3 `eval_runs_ledger` (Append-Only Immutable Evaluation Ledger)
- **Primary Key**: `run_id` (`TEXT`).
- **Columns**:
  - `total_cases` (`INTEGER NOT NULL`)
  - `passed_cases` (`INTEGER NOT NULL`)
  - `avg_faithfulness` (`REAL` / `DOUBLE PRECISION NOT NULL`)
  - `avg_citation_precision` (`REAL` / `DOUBLE PRECISION NOT NULL`)
  - `avg_context_recall` (`REAL` / `DOUBLE PRECISION NOT NULL`)
  - `avg_latency_ms` (`REAL` / `DOUBLE PRECISION NOT NULL`)
  - `passed_gate` (`INTEGER` / `BOOLEAN NOT NULL`)
  - `results_json` (`TEXT` / `JSONB NOT NULL`)
  - `created_at` (`TEXT` / `TIMESTAMPTZ NOT NULL`)
- **Immutability Triggers**:
  - `trg_eval_ledger_no_update`: Fires `BEFORE UPDATE ON eval_runs_ledger`, aborts with `APPEND_ONLY_LEDGER_VIOLATION: Ledger records are immutable`.
  - `trg_eval_ledger_no_delete`: Fires `BEFORE DELETE ON eval_runs_ledger`, aborts with `APPEND_ONLY_LEDGER_VIOLATION: Ledger records cannot be deleted`.

---

## 3. Environment Isolation Matrix

| Variable | `.env.development.example` | `.env.production.example` | Purpose |
| :--- | :--- | :--- | :--- |
| `CORTEX_ENV` | `development` | `production` | Runtime environment selector |
| `CORTEX_PORT` | `8090` | `8090` | HTTP server port for Cortex RAG API |
| `CORTEX_SQLITE_PATH` | `./.data/cortex_dev.sqlite3` | *(unused in prod)* | Local SQLite hybrid store path for zero-config dev |
| `DATABASE_URL` | `postgresql://cortex_dev:cortex_dev_local@localhost:5439/cortex_rag_dev` | `postgresql://cortex_prod_ro_rw:<SECRET>@pgvector-prod.internal:5432/cortex_rag_prod?sslmode=require` | PostgreSQL + `pgvector` connection string |

---

## 4. Migration & Rollback Strategy

- **Forward Migration**: `migrations/001_init_pgvector_hybrid.sql` initializes the `vector` extension, creates `document_chunks`, `query_audit_log`, and `eval_runs_ledger`, builds HNSW and GIN indexes, and attaches the `prevent_ledger_mutation()` trigger function to both ledger tables.
- **Rollback Discipline**: Ledger tables (`query_audit_log`, `eval_runs_ledger`) are never dropped in production without a signed compliance snapshot export. Index modifications (`idx_document_chunks_embedding_hnsw`, `idx_document_chunks_search_tsv_gin`) can be rebuilt concurrently via `DROP INDEX CONCURRENTLY` / `CREATE INDEX CONCURRENTLY`.
