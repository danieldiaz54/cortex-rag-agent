"""Application service facade for Cortex Hybrid RAG Multi-Agent operations."""

from __future__ import annotations

from typing import Any

from src.domain.agents import MultiAgentRagPipeline
from src.domain.entities import DocumentChunk
from src.domain.eval_harness import RagEvalHarness
from src.domain.exceptions import InvalidDocumentError, InvalidQueryError
from src.infrastructure.persistence.seed_corpus import (
    build_default_eval_cases,
    seed_default_corpus,
)
from src.infrastructure.persistence.sqlite_store import SQLiteHybridStore


class CortexApplicationService:
    """Coordinates MultiAgentRagPipeline, RagEvalHarness, and SQLiteHybridStore."""

    def __init__(
        self,
        store: SQLiteHybridStore | str | None = None,
        *,
        db_path: str = ":memory:",
        auto_seed: bool = True,
        pipeline: MultiAgentRagPipeline | None = None,
        eval_harness: RagEvalHarness | None = None,
    ) -> None:
        if isinstance(store, str):
            resolved_db_path = store
            self.store = SQLiteHybridStore(db_path=resolved_db_path)
        elif store is not None:
            self.store = store
        else:
            self.store = SQLiteHybridStore(db_path=db_path)

        if auto_seed and len(self.store.list_chunks()) == 0:
            seed_default_corpus(self.store)

        self.pipeline = pipeline or MultiAgentRagPipeline()
        self._chunks: tuple[DocumentChunk, ...] = tuple(self.store.list_chunks())
        self.eval_harness = eval_harness or RagEvalHarness(
            pipeline=self.pipeline,
            chunks=self._chunks,
        )

    def _refresh_corpus(self) -> tuple[DocumentChunk, ...]:
        """Refresh in-memory cached corpus snapshot from the underlying hybrid store."""
        self._chunks = tuple(self.store.list_chunks())
        self.eval_harness._default_chunks = self._chunks
        return self._chunks

    @staticmethod
    def _serialize_chunk(chunk: DocumentChunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "section": chunk.section,
            "content": chunk.content,
            "tags": list(chunk.tags),
            "embedding_dim": len(chunk.embedding),
        }

    def get_health(self) -> dict[str, Any]:
        """Return service readiness, corpus count, and ledger telemetry counts."""
        chunks = self._refresh_corpus()
        eval_runs = self.store.list_eval_reports(limit=1000)
        audits = self.store.list_query_audits(limit=1000)
        return {
            "status": "ok",
            "service": "cortex-rag-agent",
            "version": "1.0.0",
            "corpus_chunks": len(chunks),
            "eval_runs_count": len(eval_runs),
            "query_audits_count": len(audits),
            "engine": "hybrid-bm25-vector-rrf",
        }

    def list_chunks(self) -> list[dict[str, Any]]:
        """Return all indexed knowledge base chunks as JSON-serializable dictionaries."""
        chunks = self._refresh_corpus()
        return [self._serialize_chunk(chunk) for chunk in chunks]

    def ingest_document(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate payload, construct DocumentChunk, upsert into store, and refresh corpus."""
        if not isinstance(payload, dict):
            raise InvalidDocumentError("Document ingestion payload must be a JSON object.")

        chunk_id = payload.get("chunk_id", "")
        doc_id = payload.get("doc_id", "")
        title = payload.get("title", "")
        raw_section = payload.get("section", "general")
        content = payload.get("content", "")
        raw_tags = payload.get("tags", ())

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise InvalidDocumentError("Field 'chunk_id' must be a non-empty string.")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise InvalidDocumentError("Field 'doc_id' must be a non-empty string.")
        if not isinstance(title, str) or not title.strip():
            raise InvalidDocumentError("Field 'title' must be a non-empty string.")
        if not isinstance(content, str) or not content.strip():
            raise InvalidDocumentError("Field 'content' must be a non-empty string.")

        section = (
            raw_section.strip()
            if isinstance(raw_section, str) and raw_section.strip()
            else "general"
        )

        if raw_tags is None:
            tags: tuple[str, ...] = ()
        elif isinstance(raw_tags, str):
            tags = tuple(t.strip() for t in raw_tags.split(",") if t.strip())
        elif isinstance(raw_tags, (list, tuple, set)):
            tags = tuple(str(t).strip() for t in raw_tags if str(t).strip())
        else:
            raise InvalidDocumentError(
                "Field 'tags' must be a list of strings or a comma-separated string."
            )

        chunk = DocumentChunk(
            chunk_id=chunk_id.strip(),
            doc_id=doc_id.strip(),
            title=title.strip(),
            section=section,
            content=content.strip(),
            tags=tags,
        )
        saved_chunk = self.store.upsert_chunk(chunk)
        refreshed = self._refresh_corpus()
        serialized = self._serialize_chunk(saved_chunk)

        return {
            "status": "ingested",
            "chunk_id": saved_chunk.chunk_id,
            "doc_id": saved_chunk.doc_id,
            "title": saved_chunk.title,
            "section": saved_chunk.section,
            "chunk": serialized,
            "corpus_size": len(refreshed),
        }

    def execute_query(self, query: str, top_k: int = 4) -> dict[str, Any]:
        """Validate inputs, run MultiAgentRagPipeline, record audit, and return structured DTO."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1 or top_k > 10:
            raise InvalidQueryError("Parameter 'top_k' must be an integer between 1 and 10.")

        chunks = self._refresh_corpus()
        response = self.pipeline.run(query, chunks, top_k=top_k)
        audit_id = self.store.append_query_audit(response)

        intent_str = (
            response.intent.value
            if hasattr(response.intent, "value")
            else str(response.intent)
        )

        return {
            "audit_id": audit_id,
            "query": response.query,
            "intent": intent_str,
            "answer": response.answer,
            "faithfulness_score": response.faithfulness_score,
            "total_latency_ms": response.total_latency_ms,
            "citations": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "section": c.section,
                    "excerpt": c.excerpt,
                    "rrf_score": c.rrf_score,
                }
                for c in response.citations
            ],
            "candidates": [
                {
                    "chunk_id": cand.chunk.chunk_id,
                    "doc_id": cand.chunk.doc_id,
                    "title": cand.chunk.title,
                    "section": cand.chunk.section,
                    "content": cand.chunk.content,
                    "tags": list(cand.chunk.tags),
                    "chunk": {
                        "chunk_id": cand.chunk.chunk_id,
                        "doc_id": cand.chunk.doc_id,
                        "title": cand.chunk.title,
                        "section": cand.chunk.section,
                        "content": cand.chunk.content,
                        "tags": list(cand.chunk.tags),
                    },
                    "bm25_score": cand.bm25_score,
                    "vector_score": cand.vector_score,
                    "bm25_rank": cand.bm25_rank,
                    "vector_rank": cand.vector_rank,
                    "rrf_score": cand.rrf_score,
                }
                for cand in response.candidates
            ],
            "traces": [
                {
                    "agent_name": t.agent_name,
                    "action": t.action,
                    "detail": t.detail,
                    "duration_ms": t.duration_ms,
                }
                for t in response.traces
            ],
        }

    def stream_query_events(
        self,
        query: str,
        top_k: int = 4,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Execute query and produce ordered SSE event tuples: trace, token, citations, done."""
        result = self.execute_query(query=query, top_k=top_k)
        events: list[tuple[str, dict[str, Any]]] = []

        for trace in result["traces"]:
            events.append(("trace", trace))

        words = result["answer"].split(" ")
        for idx, word in enumerate(words):
            token_piece = word if idx == len(words) - 1 else f"{word} "
            events.append(
                (
                    "token",
                    {
                        "token": token_piece,
                        "text": token_piece,
                        "index": idx,
                    },
                )
            )

        events.append(
            (
                "citations",
                {
                    "citations": result["citations"],
                    "faithfulness_score": result["faithfulness_score"],
                },
            )
        )
        events.append(("done", result))
        return events

    def run_eval_suite(
        self,
        enforce_gate: bool = True,
        *,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Run RagEvalHarness over default golden cases, append to ledger, and return report."""
        if not isinstance(enforce_gate, bool):
            raise InvalidQueryError("Parameter 'enforce_gate' must be a boolean.")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1 or top_k > 10
        ):
            raise InvalidQueryError("Parameter 'top_k' must be an integer between 1 and 10.")

        chunks = self._refresh_corpus()
        cases = build_default_eval_cases()
        report = self.eval_harness.run_suite(
            cases,
            chunks=chunks,
            top_k=top_k,
            enforce_gate=enforce_gate,
        )
        self.store.append_eval_report(report)

        return {
            "run_id": report.run_id,
            "total_cases": report.total_cases,
            "passed_cases": report.passed_cases,
            "avg_faithfulness": report.avg_faithfulness,
            "avg_citation_precision": report.avg_citation_precision,
            "avg_context_recall": report.avg_context_recall,
            "avg_latency_ms": report.avg_latency_ms,
            "passed_gate": report.passed_gate,
            "results": [
                {
                    "case_id": r.case_id,
                    "query": r.query,
                    "intent": r.intent.value if hasattr(r.intent, "value") else str(r.intent),
                    "faithfulness": r.faithfulness,
                    "citation_precision": r.citation_precision,
                    "context_recall": r.context_recall,
                    "keyword_hit_rate": r.keyword_hit_rate,
                    "latency_ms": r.latency_ms,
                    "passed": r.passed,
                }
                for r in report.results
            ],
        }

    def list_eval_ledger(self, limit: int = 20) -> list[dict[str, Any]]:
        """List immutable evaluation suite reports from the append-only ledger."""
        return self.store.list_eval_reports(limit=limit)

    def list_query_audits(self, limit: int = 50) -> list[dict[str, Any]]:
        """List immutable query audit records from the append-only audit log."""
        return self.store.list_query_audits(limit=limit)
