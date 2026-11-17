# Specification: HTTP REST & Server-Sent Events (SSE) API Contracts

- **Bounded Context**: `cortex-rag-agent` — Application & Integration Layer (`src/application/`, `src/interfaces/http/`)
- **Status**: Approved (SDD-First)
- **Author Role**: `integration-architect`
- **Protocol**: HTTP/1.1 JSON REST + `text/event-stream` (SSE)

---

## 1. Architectural Overview & Boundary Invariants

The Integration Layer exposes the pure Domain Core (`src/domain/`) and Persistence Layer (`src/infrastructure/persistence/`) over deterministic HTTP/1.1 REST and Server-Sent Events (SSE) interfaces:

1. **Zero-Dependency Native Runtime (`src/interfaces/http/server.py`)**:
   - Built on Python 3.12's standard `http.server.ThreadingHTTPServer` and `BaseHTTPRequestHandler` so the entire backend and evaluation suite run natively without external third-party packages.
   - An optional FastAPI ASGI adapter (`src/interfaces/http/fastapi_app.py`) mounts the exact same `CortexApplicationService` when deployed in ASGI container environments.
2. **Request Hardening & Guardrails**:
   - Strict request body size limit of **64 KB (`65,536` bytes)** on all `POST` endpoints (`400 Bad Request` on overflow).
   - Strict JSON parsing and DTO schema validation at the boundary before invoking domain logic.
   - CORS headers (`Access-Control-Allow-Origin: *`, `Access-Control-Allow-Methods: GET, POST, OPTIONS`, `Access-Control-Allow-Headers: Content-Type, Accept`) for browser workbench compatibility.
3. **Zero Stack-Trace Leakage**:
   - Domain exceptions are deterministically mapped to semantic HTTP status codes (`400`, `404`, `422`, `500`) with structured JSON error envelopes (`{"error": {"code": "...", "message": "..."}}`). Internal tracebacks are never serialized into responses.

---

## 2. Semantic Error Mapping Contract

All error responses return `Content-Type: application/json; charset=utf-8` with the canonical envelope:

```json
{
  "error": {
    "code": "INVALID_QUERY",
    "message": "Query cannot be empty or whitespace."
  }
}
```

| Exception / Condition | Error `code` | HTTP Status | Semantic Meaning |
| :--- | :--- | :--- | :--- |
| Malformed JSON / Body > 64 KB / Missing body | `InvalidRequestError` / `INVALID_QUERY` | `400 Bad Request` | Syntactically invalid HTTP payload or parameters |
| `InvalidQueryError` | `InvalidQueryError` | `400 Bad Request` | Empty query, query > 2,000 chars, or `top_k` outside `[1, 10]` |
| `InvalidDocumentError` | `InvalidDocumentError` | `400 Bad Request` | Missing or empty `chunk_id`, `doc_id`, `title`, or `content` |
| Unknown route or HTTP method | `NotFoundError` | `404 Not Found` | Endpoint path does not exist |
| `EmptyRetrievalError` | `EmptyRetrievalError` | `422 Unprocessable Entity` | Document corpus is empty or returned zero candidates |
| `InvalidCitationError` | `InvalidCitationError` | `422 Unprocessable Entity` | Synthesized answer cited a `[cite:id]` not in retrieved candidates |
| `UngroundedClaimError` | `UngroundedClaimError` | `422 Unprocessable Entity` | Answer lacks citations or failed `faithfulness_score >= min_faithfulness` |
| `EvalRegressionError` | `EvalRegressionError` | `422 Unprocessable Entity` | Evaluation suite fell below quality gate thresholds |
| Unhandled unexpected exception | `InternalServerError` | `500 Internal Server Error` | Generic safe message (`"An unexpected internal error occurred."`) |

---

## 3. Endpoint DTO Contracts

### 3.1 `GET /` and `GET /ui` — Interactive RAG & Telemetry Workbench
- **Response (`200 OK`)**: `Content-Type: text/html; charset=utf-8` serving `ui/index.html`.

### 3.2 `GET /api/v1/health` — System & Corpus Readiness
- **Response (`200 OK`)**:
```json
{
  "status": "ok",
  "service": "cortex-rag-agent",
  "version": "1.0.0",
  "corpus_chunks": 8,
  "eval_runs_count": 0,
  "query_audits_count": 0,
  "engine": "hybrid-bm25-vector-rrf"
}
```

### 3.3 `GET /api/v1/chunks` — List Indexed Knowledge Base Chunks
- **Response (`200 OK`)**:
```json
{
  "chunks": [
    {
      "chunk_id": "chunk-rag-001",
      "doc_id": "DOC-RAG-ARCH-01",
      "title": "Hybrid Retrieval Architecture: BM25, pgvector, and Reciprocal Rank Fusion",
      "section": "retrieval-engine",
      "content": "Cortex Hybrid RAG combines Okapi BM25 lexical scoring...",
      "tags": ["bm25", "pgvector", "rrf", "hybrid-rag", "embeddings"],
      "embedding_dim": 64
    }
  ],
  "total": 8
}
```

### 3.4 `POST /api/v1/documents/ingest` — Upsert Document Chunk into Hybrid Store
- **Request Body (`application/json`)**:
```json
{
  "chunk_id": "chunk-custom-009",
  "doc_id": "DOC-ARCH-09",
  "title": "Zero-Trust Service Mesh Mutual TLS Rotation",
  "section": "security",
  "content": "SPIFFE/SPIRE workload identities rotate X.509 SVID certificates every 30 minutes.",
  "tags": ["mtls", "spiffe", "zero-trust"]
}
```
- **Response (`200 OK`)**:
```json
{
  "status": "ingested",
  "chunk_id": "chunk-custom-009",
  "doc_id": "DOC-ARCH-09",
  "title": "Zero-Trust Service Mesh Mutual TLS Rotation",
  "section": "security",
  "chunk": {
    "chunk_id": "chunk-custom-009",
    "doc_id": "DOC-ARCH-09",
    "title": "Zero-Trust Service Mesh Mutual TLS Rotation",
    "section": "security",
    "content": "SPIFFE/SPIRE workload identities rotate X.509 SVID certificates every 30 minutes.",
    "tags": ["mtls", "spiffe", "zero-trust"],
    "embedding_dim": 64
  },
  "corpus_size": 9
}
```

### 3.5 `POST /api/v1/rag/query` — Execute Multi-Agent Hybrid RAG Pipeline
- **Request Body (`application/json`)**:
```json
{
  "query": "How does ERR-409-OPTIMISTIC-LOCK happen in distributed ledgers and how is it resolved?",
  "top_k": 4
}
```
- **Response (`200 OK`)**:
```json
{
  "audit_id": "aud-4f8a9c12e3b0",
  "query": "How does ERR-409-OPTIMISTIC-LOCK happen in distributed ledgers and how is it resolved?",
  "intent": "LEXICAL_EXACT",
  "answer": "Concurrency Resolution for ERR-409-OPTIMISTIC-LOCK in Distributed Ledgers (error-catalog): Error code ERR-409-OPTIMISTIC-LOCK occurs when concurrent ledger writers conflict on aggregate sequence numbers, requiring SHA-256 idempotency key verification and exponential jitter backoff. [cite:chunk-err-409]",
  "faithfulness_score": 1.0,
  "total_latency_ms": 1.84,
  "citations": [
    {
      "chunk_id": "chunk-err-409",
      "doc_id": "DOC-LEDGER-02",
      "title": "Concurrency Resolution for ERR-409-OPTIMISTIC-LOCK in Distributed Ledgers",
      "section": "error-catalog",
      "excerpt": "Error code ERR-409-OPTIMISTIC-LOCK occurs when concurrent ledger writers conflict...",
      "rrf_score": 0.01639344
    }
  ],
  "candidates": [
    {
      "chunk_id": "chunk-err-409",
      "doc_id": "DOC-LEDGER-02",
      "title": "Concurrency Resolution for ERR-409-OPTIMISTIC-LOCK in Distributed Ledgers",
      "section": "error-catalog",
      "content": "Error code ERR-409-OPTIMISTIC-LOCK occurs...",
      "tags": ["err-409-optimistic-lock", "err-409", "optimistic-locking", "idempotency", "ledger"],
      "bm25_score": 11.4218,
      "vector_score": 0.7812,
      "bm25_rank": 1,
      "vector_rank": 1,
      "rrf_score": 0.01639344
    }
  ],
  "traces": [
    {
      "agent_name": "RouterAgent",
      "action": "classify_intent",
      "detail": "intent=LEXICAL_EXACT (bm25_weight=0.75, vector_weight=0.25)",
      "duration_ms": 0.12
    }
  ]
}
```

### 3.6 `POST /api/v1/rag/stream` (and `GET /api/v1/rag/stream?query=...&top_k=4`) — SSE Streaming Pipeline
- **Response (`200 OK`, `Content-Type: text/event-stream; charset=utf-8`)**:
  Emits ordered SSE frames:
  1. `event: trace` — One event per agent step (`RouterAgent`, `HybridRetrieverAgent`, `SynthesizerAgent`, `CitationGuardrailAgent`).
  2. `event: token` — Incremental token stream (`{"token": "...", "index": 0}`).
  3. `event: citations` — Grounded citation list and `faithfulness_score`.
  4. `event: done` — Full `RagResponse` DTO with `audit_id`.

### 3.7 `POST /api/v1/evals/run` — Execute Golden Dataset Evaluation Suite
- **Request Body (`application/json`, optional)**:
```json
{
  "enforce_gate": true
}
```
- **Response (`200 OK`)**:
```json
{
  "run_id": "eval-9a1b2c3d4e5f",
  "total_cases": 5,
  "passed_cases": 5,
  "avg_faithfulness": 0.985,
  "avg_citation_precision": 1.0,
  "avg_context_recall": 1.0,
  "avg_latency_ms": 1.52,
  "passed_gate": true,
  "results": [
    {
      "case_id": "eval-001",
      "query": "How does ERR-409-OPTIMISTIC-LOCK happen in distributed ledgers and how is it resolved?",
      "intent": "LEXICAL_EXACT",
      "faithfulness": 1.0,
      "citation_precision": 1.0,
      "context_recall": 1.0,
      "keyword_hit_rate": 1.0,
      "latency_ms": 1.41,
      "passed": true
    }
  ]
}
```

### 3.8 `GET /api/v1/evals/ledger` — List Append-Only Evaluation Runs
- **Response (`200 OK`)**:
```json
{
  "ledger": [
    {
      "run_id": "eval-9a1b2c3d4e5f",
      "total_cases": 5,
      "passed_cases": 5,
      "avg_faithfulness": 0.985,
      "avg_citation_precision": 1.0,
      "avg_context_recall": 1.0,
      "avg_latency_ms": 1.52,
      "passed_gate": true,
      "results": [],
      "created_at": "2026-11-17T19:20:00+00:00"
    }
  ],
  "runs": [],
  "total": 1
}
```

### 3.9 `GET /api/v1/audit/queries` — List Append-Only Query Audit Log
- **Response (`200 OK`)**:
```json
{
  "audits": [
    {
      "audit_id": "aud-4f8a9c12e3b0",
      "query": "How does ERR-409-OPTIMISTIC-LOCK happen in distributed ledgers and how is it resolved?",
      "intent": "LEXICAL_EXACT",
      "answer": "...",
      "citations": [],
      "faithfulness_score": 1.0,
      "latency_ms": 1.84,
      "created_at": "2026-11-17T19:20:00+00:00"
    }
  ],
  "queries": [],
  "total": 1
}
```
