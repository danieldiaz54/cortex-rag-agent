"""Default technical documentation seed corpus and golden evaluation dataset for Cortex RAG."""

from __future__ import annotations

from typing import Any

from src.domain.entities import DocumentChunk, EvalCase
from src.infrastructure.persistence.sqlite_store import SQLiteHybridStore

DEFAULT_TECHNICAL_CHUNKS: list[dict[str, Any]] = [
    {
        "chunk_id": "chunk-rag-001",
        "doc_id": "DOC-RAG-ARCH-01",
        "title": "Hybrid Retrieval Architecture: BM25, pgvector, and Reciprocal Rank Fusion",
        "section": "retrieval-engine",
        "content": (
            "Cortex Hybrid RAG combines Okapi BM25 lexical scoring (k1=1.5, b=0.75) with "
            "64-dimensional L2-normalized pgvector dense cosine similarity using weighted "
            "Reciprocal Rank Fusion (RRF, k=60). Exact technical queries prioritize BM25 "
            "lexical matching while conceptual architecture queries weight pgvector dense "
            "embeddings higher through dynamic RRF fusion."
        ),
        "tags": ["bm25", "pgvector", "rrf", "hybrid-rag", "embeddings"],
    },
    {
        "chunk_id": "chunk-err-409",
        "doc_id": "DOC-LEDGER-02",
        "title": "Concurrency Resolution for ERR-409-OPTIMISTIC-LOCK in Distributed Ledgers",
        "section": "error-catalog",
        "content": (
            "Error code ERR-409-OPTIMISTIC-LOCK occurs when concurrent ledger writers conflict "
            "on aggregate sequence numbers, requiring SHA-256 idempotency key verification and "
            "exponential jitter backoff. Clients remediate ERR-409-OPTIMISTIC-LOCK by refreshing "
            "the latest aggregate version snapshot, validating idempotency state, and retrying "
            "with bounded backoff."
        ),
        "tags": ["err-409-optimistic-lock", "err-409", "optimistic-locking", "idempotency", "ledger"],
    },
    {
        "chunk_id": "chunk-aga8-003",
        "doc_id": "DOC-SCADA-03",
        "title": "AGA-8 Supercritical Compressibility and Newton-Raphson DivergenceException",
        "section": "thermodynamics",
        "content": (
            "During high-pressure custody transfer above 230 bar, the AGA-8 Gross Characterization "
            "solver raises DivergenceException when the Newton-Raphson compressibility root-finder "
            "exceeds 50 iterations. Operators prevent AGA-8 DivergenceException at 230 bar by "
            "clamping reduced density bounds and switching to the GERG-2008 fallback equation of state."
        ),
        "tags": ["aga-8", "newton-raphson", "divergenceexception", "custody-transfer", "scada"],
    },
    {
        "chunk_id": "chunk-outbox-004",
        "doc_id": "DOC-KAFKA-04",
        "title": "Transactional Outbox Pattern with Kafka and Idempotent Consumers",
        "section": "event-streaming",
        "content": (
            "The Transactional Outbox pattern guarantees atomic delivery of domain state mutations "
            "and outbound Kafka integration events within a single ACID database transaction. "
            "An asynchronous Transactional Outbox relay polls uncommitted rows in creation order "
            "and publishes to Kafka partitions while preserving ACID durability and idempotency."
        ),
        "tags": ["transactional-outbox", "kafka", "cdc", "idempotency", "append-only"],
    },
    {
        "chunk_id": "chunk-raft-005",
        "doc_id": "DOC-DIST-05",
        "title": "Raft Consensus Quorum Leases and Split-Brain Fencing Tokens",
        "section": "consensus",
        "content": (
            "Distributed storage clusters prevent split-brain writes during network partitions "
            "by requiring a strict Raft majority quorum and monotonically increasing epoch "
            "fencing tokens. Any isolated Raft leader whose quorum lease expires or whose "
            "fencing token is superseded is immediately rejected by storage nodes."
        ),
        "tags": ["raft", "quorum", "fencing-token", "split-brain", "distributed-systems"],
    },
    {
        "chunk_id": "chunk-otel-006",
        "doc_id": "DOC-OBS-06",
        "title": "OpenTelemetry W3C TraceContext Propagation Across Multi-Agent Pipelines",
        "section": "observability",
        "content": (
            "Every multi-agent orchestration request propagates OpenTelemetry W3C TraceContext "
            "headers (traceparent and tracestate) across RouterAgent, HybridRetrieverAgent, "
            "SynthesizerAgent, and CitationGuardrailAgent spans. OpenTelemetry span attributes "
            "capture per-agent latency_ms, RRF candidate scores, and faithfulness verification status."
        ),
        "tags": ["opentelemetry", "traceparent", "w3c-tracecontext", "multi-agent", "telemetry"],
    },
    {
        "chunk_id": "chunk-sqlite-007",
        "doc_id": "DOC-MOBILE-07",
        "title": "Offline-First SQLite Sync Queues and Causal Lamport Clocks",
        "section": "edge-sync",
        "content": (
            "Field edge clients persist telemetry mutations in an append-only Offline-First "
            "SQLite sync queue stamped with monotonic Lamport clocks to preserve causal ordering "
            "during reconnection. Upon network recovery, the Offline-First SQLite engine drains "
            "pending mutations in causal Lamport order and resolves conflicts deterministically."
        ),
        "tags": ["sqlite", "offline-first", "sync-queue", "lamport-clock", "crdt"],
    },
    {
        "chunk_id": "chunk-guard-008",
        "doc_id": "DOC-RAG-GUARD-08",
        "title": "Citation Guardrails, Grounding Verification, and Append-Only Audit Ledgers",
        "section": "guardrails",
        "content": (
            "CitationGuardrailAgent enforces strict grounding by verifying that every synthesized "
            "claim carries a valid [cite:chunk_id] marker matching retrieved evidence and achieves "
            "a faithfulness_score of at least 0.65. Verified responses are recorded in the "
            "immutable append-only query_audit_log protected by database triggers against UPDATE or DELETE."
        ),
        "tags": ["citation-guardrail", "faithfulness", "ungroundedclaimerror", "append-only", "audit-ledger"],
    },
]

DEFAULT_EVAL_CASES: list[dict[str, Any]] = [
    {
        "case_id": "eval-001",
        "query": "How does ERR-409-OPTIMISTIC-LOCK happen in distributed ledgers and how is it resolved?",
        "expected_chunk_ids": ["chunk-err-409"],
        "expected_keywords": ["ERR-409-OPTIMISTIC-LOCK", "idempotency", "backoff"],
    },
    {
        "case_id": "eval-002",
        "query": "Why does AGA-8 raise DivergenceException at 230 bar in the Newton-Raphson solver?",
        "expected_chunk_ids": ["chunk-aga8-003"],
        "expected_keywords": ["AGA-8", "DivergenceException", "230"],
    },
    {
        "case_id": "eval-003",
        "query": "Explain how Cortex Hybrid RAG combines BM25, pgvector, and Reciprocal Rank Fusion (RRF).",
        "expected_chunk_ids": ["chunk-rag-001"],
        "expected_keywords": ["BM25", "pgvector", "RRF"],
    },
    {
        "case_id": "eval-004",
        "query": "How does the Transactional Outbox pattern guarantee atomic event delivery to Kafka?",
        "expected_chunk_ids": ["chunk-outbox-004"],
        "expected_keywords": ["Outbox", "Kafka", "ACID"],
    },
    {
        "case_id": "eval-005",
        "query": "How do Offline-First SQLite sync queues preserve causal ordering on edge devices?",
        "expected_chunk_ids": ["chunk-sqlite-007"],
        "expected_keywords": ["SQLite", "Lamport", "causal"],
    },
]


def build_default_eval_cases() -> tuple[EvalCase, ...]:
    """Instantiate domain EvalCase value objects from DEFAULT_EVAL_CASES."""
    return tuple(
        EvalCase(
            case_id=str(item["case_id"]),
            query=str(item["query"]),
            expected_chunk_ids=tuple(str(cid) for cid in item["expected_chunk_ids"]),
            expected_keywords=tuple(str(kw) for kw in item["expected_keywords"]),
        )
        for item in DEFAULT_EVAL_CASES
    )


def seed_default_corpus(store: SQLiteHybridStore) -> int:
    """Seed the SQLiteHybridStore with the 8 default technical chunks if not already populated."""
    count = 0
    for raw in DEFAULT_TECHNICAL_CHUNKS:
        chunk = DocumentChunk(
            chunk_id=str(raw["chunk_id"]),
            doc_id=str(raw["doc_id"]),
            title=str(raw["title"]),
            section=str(raw["section"]),
            content=str(raw["content"]),
            tags=tuple(str(t) for t in raw.get("tags", ())),
        )
        store.upsert_chunk(chunk)
        count += 1
    return count
